
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

#include "lodepng.h" //Used for png encoding.
#include "renderdoc.h"

static int WIDTH;
static int HEIGHT;
static int DEPTH;

static int WORKGROUP_SIZE_X;
static int WORKGROUP_SIZE_Y;
static int WORKGROUP_SIZE_Z;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

/*
The application launches a compute shader that renders the mandelbrot set,
by rendering it into a storage buffer.
The storage buffer is then read from the GPU, and saved as .png. 
*/
class ComputeApplication {
private:
    // The pixels of the rendered mandelbrot set are in this format:
    struct uvec4 {
        uint32_t x, y, z, w;
    };
    struct Pixel {
        float r, g, b, a;
        uvec4 numWorkGroups;
        uvec4 workGroupSize;
        uvec4 workGroupID;
        uvec4 localInvocationID;
        uvec4 globalInvocationID;
        uvec4 localInvocationIndex;
        uvec4 subgroup;
    };
    
    /*
    In order to use Vulkan, you must create an instance. 
    */
    VkInstance instance;

    VkDebugReportCallbackEXT debugReportCallback;
    /*
    The physical device is some device on the system that supports usage of Vulkan.
    Often, it is simply a graphics card that supports Vulkan. 
    */
    VkPhysicalDevice physicalDevice;
    /*
    Then we have the logical device VkDevice, which basically allows 
    us to interact with the physical device. 
    */
    VkDevice device;

    /*
    The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.

    We will be creating a simple compute pipeline in this application. 
    */
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    /*
    The command buffer is used to record commands, that will be submitted to a queue.

    To allocate such command buffers, we use a command pool.
    */
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffers[2];

    /*

    Descriptors represent resources in shaders. They allow us to use things like
    uniform buffers, storage buffers and images in GLSL. 

    A single descriptor represents a single resource, and several descriptors are organized
    into descriptor sets, which are basically just collections of descriptors.
    */
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    /*
    The mandelbrot set will be rendered to this buffer.

    The memory that backs the buffer is bufferMemory. 
    */
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
        
    uint32_t bufferSize; // size of `buffer` in bytes.

    std::vector<const char *> enabledLayers;

    /*
    In order to execute commands on a device(GPU), the commands must be submitted
    to a queue. The commands are stored in a command buffer, and this command buffer
    is given to the queue. 

    There will be different kinds of queues on the device. Not all queues support
    graphics operations, for instance. For this application, we at least want a queue
    that supports compute operations. 
    */
    VkQueue queue; // a queue supporting compute operations.

    /*
    Groups of queues that have the same capabilities(for instance, they all supports graphics and computer operations),
    are grouped into queue families. 
    
    When submitting a command buffer, you must specify to which queue in the family you are submitting to. 
    This variable keeps track of the index of that queue in its family. 
    */
    uint32_t queueFamilyIndex;

    struct {
        bool enabled;
        std::vector<VkPerformanceCounterStorageKHR> storages;
        std::vector<uint32_t> selectedCounters;
        VkQueryPoolPerformanceCreateInfoKHR performanceQueryCreateInfo;
        uint32_t numPasses;
        VkQueryPool queryPoolKHR;
        VkQueryPool queryPoolPipeline;

        unsigned EUThreadOccupaccyIdx;
        unsigned GPUTimeElapsedIdx;
        unsigned CSThreadsDispatchedIdx;
        bool show_csv;
    } perf;

public:
    void run() {
        perf.enabled = getenv("PERF_ENABLED") == NULL;
        perf.show_csv = getenv("CSV") != NULL;
        FILE *statsFile = NULL;

        if (perf.enabled && perf.show_csv) {
            statsFile = fopen("stats.csv", "w");
            if (!statsFile) {
                perror("fopen stats.csv");
                exit(2);
            }
            fprintf(statsFile, "x:int,y:int,z:int,time_ms:int,threads:int,invocations:int,simd:int,thread_occupancy_pct:int\n");
        }

        RENDERDOC_API_1_4_1 *rdoc_api = NULL;

        void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (mod) {
            pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
            int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_1, (void **)&rdoc_api);
            assert(ret == 1);
        }

        // Buffer size of the storage buffer that will contain the rendered mandelbrot set.
        bufferSize = sizeof(Pixel) * WIDTH * HEIGHT * DEPTH;

        // Initialize vulkan:
        createInstance();
        findPhysicalDevice();
        if (perf.enabled)
            selectPerfCounters();
        createDevice();
        if (perf.enabled)
            createQueries();
        if (rdoc_api)
            rdoc_api->StartFrameCapture(NULL, NULL);
        createBuffer();
        createDescriptorSetLayout();
        createDescriptorSet();
        createComputePipeline();

        if (perf.enabled) {
            VkAcquireProfilingLockInfoKHR lockInfo;
            lockInfo.sType = VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR;
            lockInfo.pNext = NULL;
            lockInfo.flags  = 0;
            lockInfo.timeout = UINT64_MAX;

            PFN_vkAcquireProfilingLockKHR vkAcquireProfilingLockKHR =
                        (PFN_vkAcquireProfilingLockKHR)
                        vkGetInstanceProcAddr(instance, "vkAcquireProfilingLockKHR");
            assert(vkAcquireProfilingLockKHR != NULL);

            VK_CHECK_RESULT(vkAcquireProfilingLockKHR(device, &lockInfo));
        }

        createCommandPool();
        allocateCommandBuffers();
        if (perf.enabled)
            createResetCommandBuffer();
        createCommandBuffer();

        if (perf.enabled) {
            // Finally, run the recorded command buffer.
            for (uint32_t counterPass = 0; counterPass < perf.numPasses; counterPass++) {
              VkPerformanceQuerySubmitInfoKHR performanceQuerySubmitInfo;
              performanceQuerySubmitInfo.sType = VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR;
              performanceQuerySubmitInfo.pNext = NULL;
              performanceQuerySubmitInfo.counterPassIndex = counterPass;

              runCommandBuffer(&commandBuffers[0], NULL);
              runCommandBuffer(&commandBuffers[1], &performanceQuerySubmitInfo);
            }

            PFN_vkReleaseProfilingLockKHR vkReleaseProfilingLockKHR =
                        (PFN_vkReleaseProfilingLockKHR)
                        vkGetInstanceProcAddr(instance, "vkReleaseProfilingLockKHR");
            assert(vkReleaseProfilingLockKHR != NULL);

            vkReleaseProfilingLockKHR(device);

            size_t cntrs = perf.selectedCounters.size();
            std::vector<VkPerformanceCounterResultKHR> recordedCounters(cntrs);

            VK_CHECK_RESULT(vkGetQueryPoolResults(device, perf.queryPoolKHR, 0, 1,
                    sizeof(VkPerformanceCounterResultKHR) * cntrs,
                    recordedCounters.data(),
                    sizeof(VkPerformanceCounterResultKHR),
                    0));
            if (0) {
                int i = 0;
                for (auto c : recordedCounters) {
                    printf("counter: %d, value: ", i);
                    switch(perf.storages[i]) {
                    case VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR:
                        printf("%d\n", c.int32);
                        break;
                    case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR:
                        printf("%u\n", c.uint32);
                        break;
                    case VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR:
                        printf("%ld\n", c.int64);
                        break;
                    case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR:
                        printf("%lu\n", c.uint64);
                        break;
                    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR:
                        printf("%f\n", c.float32);
                        break;
                    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR:
                        printf("%g\n", c.float64);
                        break;
                    }
                    i++;
                }
            }

            std::vector<uint64_t> recordedCountersPipeline(1);
            VK_CHECK_RESULT(vkGetQueryPoolResults(device, perf.queryPoolPipeline, 0, 1,
                    sizeof(uint64_t) * 1,
                    recordedCountersPipeline.data(),
                    sizeof(uint64_t),
                    0));

            if (perf.show_csv) {
                fprintf(statsFile, "%d,%d,%d,", WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z);
                fprintf(statsFile, "%d,", (int)(recordedCounters[perf.GPUTimeElapsedIdx].uint64/1000000.0));
                fprintf(statsFile, "%lu,", recordedCounters[perf.CSThreadsDispatchedIdx].uint64);
                fprintf(statsFile, "%lu,", recordedCountersPipeline[0]);
                fprintf(statsFile, "%lu,", recordedCountersPipeline[0] / recordedCounters[perf.CSThreadsDispatchedIdx].uint64);
                fprintf(statsFile, "%d\n", (int)(recordedCounters[perf.EUThreadOccupaccyIdx].float32));
            } else {
                printf("EU Thread Occupancy:   %f %%\n", recordedCounters[perf.EUThreadOccupaccyIdx].float32);
                printf("CS Threads Dispatched: %lu\n", recordedCounters[perf.CSThreadsDispatchedIdx].uint64);
                if (0)
                    printf("GPU Time Elapsed:      %lu ns\n", recordedCounters[perf.GPUTimeElapsedIdx].uint64);
                printf("GPU Time Elapsed:      %f ms\n", recordedCounters[perf.GPUTimeElapsedIdx].uint64/1000000.0);
                printf("CS Invocations:        %lu\n", recordedCountersPipeline[0]);
            }

        } else {
            runCommandBuffer(&commandBuffers[1], NULL);
        }

        // The former command rendered a mandelbrot set to a buffer.
        // Save that buffer as a png on disk.
        saveRenderedImage();
        if (rdoc_api)
            rdoc_api->EndFrameCapture(NULL, NULL);

        // Clean up all vulkan resources.
        cleanup();
    }

    void saveRenderedImage() {
        void* mappedMemory = NULL;
        // Map the buffer memory, so that we can read from it on the CPU.
        vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedMemory);
        Pixel* pmappedMemory = (Pixel *)mappedMemory;

        // Get the color data from the buffer, and cast it to bytes.
        // We save the data to a vector.
        std::vector<unsigned char> image;
        image.reserve(WIDTH * HEIGHT * DEPTH * 4);

        FILE *dataFile = fopen("data.csv", "w");

        fprintf(dataFile, "z:int,");
        fprintf(dataFile, "GIID.z:int,");

        fprintf(dataFile, "y:int,");
        fprintf(dataFile, "GIID.y:int,");

        fprintf(dataFile, "x:int,");
        fprintf(dataFile, "GIID.x:int,");

        fprintf(dataFile, "WGID.z:int,");
        fprintf(dataFile, "NumWG.z:int,");

        fprintf(dataFile, "WGID.y:int,");
        fprintf(dataFile, "NumWG.y:int,");

        fprintf(dataFile, "WGID.x:int,");
        fprintf(dataFile, "NumWG.x:int,");

        fprintf(dataFile, "LIID.z:int,");
        fprintf(dataFile, "WGS.z:int,");

        fprintf(dataFile, "LIID.y:int,");
        fprintf(dataFile, "WGS.y:int,");

        fprintf(dataFile, "LIID.x:int,");
        fprintf(dataFile, "WGS.x:int,");

        fprintf(dataFile, "LIIndex:int,");

        fprintf(dataFile, "SGID:int,");
        fprintf(dataFile, "NumSG:int,");

        fprintf(dataFile, "SGIID:int,");
        fprintf(dataFile, "SGS:int,");

        fprintf(dataFile, "rFloat:string,");
        fprintf(dataFile, "rChar:int,");
        fprintf(dataFile, "gFloat:string,");
        fprintf(dataFile, "gChar:int,");
        fprintf(dataFile, "bFloat:string,");
        fprintf(dataFile, "bChar:int,");
        fprintf(dataFile, "aFloat:string,");
        fprintf(dataFile, "aChar:int\n");

        for (int i = 0; i < WIDTH * HEIGHT * DEPTH; ++i) {
            fprintf(dataFile, "%u,", i  / (WIDTH * HEIGHT));
            fprintf(dataFile, "%u,", pmappedMemory[i].globalInvocationID.z);

            fprintf(dataFile, "%u,", (i % (WIDTH * HEIGHT)) / WIDTH);
            fprintf(dataFile, "%u,", pmappedMemory[i].globalInvocationID.y);

            fprintf(dataFile, "%u,", (i % (WIDTH * HEIGHT)) % WIDTH);
            fprintf(dataFile, "%u,", pmappedMemory[i].globalInvocationID.x);

            fprintf(dataFile, "%u,", pmappedMemory[i].workGroupID.z);
            fprintf(dataFile, "%u,", pmappedMemory[i].numWorkGroups.z);

            fprintf(dataFile, "%u,", pmappedMemory[i].workGroupID.y);
            fprintf(dataFile, "%u,", pmappedMemory[i].numWorkGroups.y);

            fprintf(dataFile, "%u,", pmappedMemory[i].workGroupID.x);
            fprintf(dataFile, "%u,", pmappedMemory[i].numWorkGroups.x);

            fprintf(dataFile, "%u,", pmappedMemory[i].localInvocationID.z);
            fprintf(dataFile, "%u,", pmappedMemory[i].workGroupSize.z);

            fprintf(dataFile, "%u,", pmappedMemory[i].localInvocationID.y);
            fprintf(dataFile, "%u,", pmappedMemory[i].workGroupSize.y);

            fprintf(dataFile, "%u,", pmappedMemory[i].localInvocationID.x);
            fprintf(dataFile, "%u,", pmappedMemory[i].workGroupSize.x);

            fprintf(dataFile, "%u,", pmappedMemory[i].localInvocationIndex.x);

            fprintf(dataFile, "%u,", pmappedMemory[i].subgroup.x); // SGID
            fprintf(dataFile, "%u,", pmappedMemory[i].subgroup.w); // NumSG

            fprintf(dataFile, "%u,", pmappedMemory[i].subgroup.y); // SGIID
            fprintf(dataFile, "%u,", pmappedMemory[i].subgroup.z); // SGS

            fprintf(dataFile, "%f,", pmappedMemory[i].r);
            fprintf(dataFile, "%u,", (unsigned char)(255.0f * (pmappedMemory[i].r)));
            fprintf(dataFile, "%f,", pmappedMemory[i].g);
            fprintf(dataFile, "%u,", (unsigned char)(255.0f * (pmappedMemory[i].g)));
            fprintf(dataFile, "%f,", pmappedMemory[i].b);
            fprintf(dataFile, "%u,", (unsigned char)(255.0f * (pmappedMemory[i].b)));
            fprintf(dataFile, "%f,", pmappedMemory[i].a);
            fprintf(dataFile, "%u", (unsigned char)(255.0f * (pmappedMemory[i].a)));

            fprintf(dataFile, "\n");

            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].r)));
            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].g)));
            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].b)));
            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].a)));
        }
        fclose(dataFile);
        // Done reading, so unmap.
        vkUnmapMemory(device, bufferMemory);

        // Now we save the acquired color data to a .png.
        unsigned error = lodepng::encode("mandelbrot.png", image, WIDTH, HEIGHT * DEPTH);
        if (error) printf("encoder error %d: %s", error, lodepng_error_text(error));
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData) {

        printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

        return VK_FALSE;
     }

    void createInstance() {
        std::vector<const char *> enabledExtensions;

        /*
        By enabling validation layers, Vulkan will emit warnings if the API
        is used incorrectly. We shall enable the layer VK_LAYER_LUNARG_standard_validation,
        which is basically a collection of several useful validation layers.
        */
        if (enableValidationLayers) {
            /*
            We get all supported layers with vkEnumerateInstanceLayerProperties.
            */
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, NULL);

            std::vector<VkLayerProperties> layerProperties(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

            /*
            And then we simply check if VK_LAYER_LUNARG_standard_validation is among the supported layers.
            */
            bool foundLayer = false;
            for (VkLayerProperties prop : layerProperties) {
                
                if (strcmp("VK_LAYER_KHRONOS_validation", prop.layerName) == 0) {
                    foundLayer = true;
                    break;
                }

            }
            
            if (!foundLayer) {
                throw std::runtime_error("Layer VK_LAYER_KHRONOS_validation not supported\n");
            }
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation"); // Alright, we can use this layer.

            /*
            We need to enable an extension named VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
            in order to be able to print the warnings emitted by the validation layer.

            So again, we just check if the extension is among the supported extensions.
            */
            
            uint32_t extensionCount;
            
            vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
            std::vector<VkExtensionProperties> extensionProperties(extensionCount);
            vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties.data());

            bool foundExtension = false;
            for (VkExtensionProperties prop : extensionProperties) {
                if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) {
                    foundExtension = true;
                    break;
                }

            }

            if (!foundExtension) {
                throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
            }
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }

        if (perf.enabled)
            enabledExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        /*
        Next, we actually create the instance.
        
        */
        
        /*
        Contains application info. This is actually not that important.
        The only real important field is apiVersion.
        */
        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Hello world app";
        applicationInfo.applicationVersion = 0;
        applicationInfo.pEngineName = "awesomeengine";
        applicationInfo.engineVersion = 0;
        applicationInfo.apiVersion = VK_API_VERSION_1_2;
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.flags = 0;
        createInfo.pApplicationInfo = &applicationInfo;
        
        // Give our desired layers and extensions to vulkan.
        createInfo.enabledLayerCount = enabledLayers.size();
        createInfo.ppEnabledLayerNames = enabledLayers.data();
        createInfo.enabledExtensionCount = enabledExtensions.size();
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    
        /*
        Actually create the instance.
        Having created the instance, we can actually start using vulkan.
        */
        VK_CHECK_RESULT(vkCreateInstance(
            &createInfo,
            NULL,
            &instance));

        /*
        Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings emitted from the validation
        layer are actually printed.
        */
        if (enableValidationLayers) {
            VkDebugReportCallbackCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
            createInfo.pfnCallback = &debugReportCallbackFn;

            // We have to explicitly load this function.
            auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
            if (vkCreateDebugReportCallbackEXT == nullptr) {
                throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
            }

            // Create and register callback.
            VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &createInfo, NULL, &debugReportCallback));
        }
    
    }

    void findPhysicalDevice() {
        /*
        In this function, we find a physical device that can be used with Vulkan.
        */

        /*
        So, first we will list all physical devices on the system with vkEnumeratePhysicalDevices .
        */
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (deviceCount == 0) {
            throw std::runtime_error("could not find a device with vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        /*
        Next, we choose a device that can be used for our purposes. 

        With VkPhysicalDeviceFeatures(), we can retrieve a fine-grained list of physical features supported by the device.
        However, in this demo, we are simply launching a simple compute shader, and there are no 
        special physical features demanded for this task.

        With VkPhysicalDeviceProperties(), we can obtain a list of physical device properties. Most importantly,
        we obtain a list of physical device limitations. For this application, we launch a compute shader,
        and the maximum size of the workgroups and total number of compute shader invocations is limited by the physical device,
        and we should ensure that the limitations named maxComputeWorkGroupCount, maxComputeWorkGroupInvocations and 
        maxComputeWorkGroupSize are not exceeded by our application.  Moreover, we are using a storage buffer in the compute shader,
        and we should ensure that it is not larger than the device can handle, by checking the limitation maxStorageBufferRange. 

        However, in our application, the workgroup size and total number of shader invocations is relatively small, and the storage buffer is
        not that large, and thus a vast majority of devices will be able to handle it. This can be verified by looking at some devices at_
        http://vulkan.gpuinfo.org/

        Therefore, to keep things simple and clean, we will not perform any such checks here, and just pick the first physical
        device in the list. But in a real and serious application, those limitations should certainly be taken into account.

        */
        for (VkPhysicalDevice device : devices) {
            if (true) { // As above stated, we do no feature checks, so just accept.
                physicalDevice = device;
                break;
            }
        }
    }

    void selectPerfCounters() {
        PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR
                vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR =
                    (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)
                    vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
        assert(vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR != NULL);

        uint32_t counterCount;

        // Get the count of counters supported
        VK_CHECK_RESULT(vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
          physicalDevice,
          queueFamilyIndex,
          &counterCount,
          NULL,
          NULL));

        std::vector<VkPerformanceCounterKHR> counters(counterCount);
        std::vector<VkPerformanceCounterDescriptionKHR> counterDescriptions(counterCount);

        for (auto &c : counters) {
            c.sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR;
            c.pNext = nullptr;
        }

        for (auto &c : counterDescriptions) {
            c.sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR;
            c.pNext = nullptr;
        }

        // Get the counters supported
        VK_CHECK_RESULT(vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
          physicalDevice,
          queueFamilyIndex,
          &counterCount,
          counters.data(),
          counterDescriptions.data()));

        unsigned idx = 0;
        for (uint32_t i = 0; i < counterDescriptions.size(); ++i) {
            const auto &c = counterDescriptions[i];
            if (strcmp(c.name, "EU Thread Occupancy") == 0) {
                perf.EUThreadOccupaccyIdx = idx;
                assert(counters[i].storage == VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR);
            } else if (strcmp(c.name, "CS Threads Dispatched") == 0) {
                perf.CSThreadsDispatchedIdx = idx;
                assert(counters[i].storage == VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR);
            } else if (strcmp(c.name, "GPU Time Elapsed") == 0) {
                perf.GPUTimeElapsedIdx = idx;
                assert(counters[i].storage == VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR);
            }

            if (strcmp(c.name, "EU Thread Occupancy") == 0 ||
                    strcmp(c.name, "CS Threads Dispatched") == 0 ||
                    strcmp(c.name, "GPU Time Elapsed") == 0) {
                if (0)
                    printf("found counter %u %s, type: %d\n", i, c.name, counters[i].storage);
                perf.selectedCounters.push_back(i);
                perf.storages.push_back(counters[i].storage);
                idx++;
            }
        }
        assert(perf.selectedCounters.size() == 3);

        perf.performanceQueryCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR;
        perf.performanceQueryCreateInfo.pNext = NULL;
        perf.performanceQueryCreateInfo.queueFamilyIndex = queueFamilyIndex;
        perf.performanceQueryCreateInfo.counterIndexCount = (uint32_t)perf.selectedCounters.size();
        perf.performanceQueryCreateInfo.pCounterIndices = perf.selectedCounters.data();

        PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR
        vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR =
                    (PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)
                    vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");
        assert(vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR != NULL);

        vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
          physicalDevice,
          &perf.performanceQueryCreateInfo,
          &perf.numPasses);
        if (0)
            printf("numPasses: %u\n", perf.numPasses);
    }


    // Returns the index of a queue family that supports compute operations. 
    uint32_t getComputeQueueFamilyIndex() {
        uint32_t queueFamilyCount;

        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

        // Retrieve all queue families.
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        // Now find a family that supports compute.
        uint32_t i = 0;
        for (; i < queueFamilies.size(); ++i) {
            VkQueueFamilyProperties props = queueFamilies[i];

            if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                // found a queue with compute. We're done!
                break;
            }
        }

        if (i == queueFamilies.size()) {
            throw std::runtime_error("could not find a queue family that supports operations");
        }

        return i;
    }

    void createDevice() {
        if (0) {
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &extensionCount, NULL);
            std::vector<VkExtensionProperties> extensionProperties(extensionCount);
            vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &extensionCount, extensionProperties.data());

            bool foundExtension = false;
            for (VkExtensionProperties prop : extensionProperties)
                printf("phys dev ext name: %s\n", prop.extensionName);
        }

#if 0
        VkPhysicalDeviceSubgroupProperties subgroupProperties;
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProperties.pNext = NULL;

        VkPhysicalDeviceProperties2 physicalDeviceProperties;
        physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        physicalDeviceProperties.pNext = &subgroupProperties;

        vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties);
        printf("sup ops: %u, sup stages: %u, sizes:%u\n",
                subgroupProperties.supportedOperations,
                subgroupProperties.supportedStages,
                subgroupProperties.subgroupSize);
#endif

        /*
        We create the logical device in this function.
        */

        /*
        When creating the device, we also specify what queues it has.
        */
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
        float queuePriorities = 1.0;  // we only have one queue, so this is not that imporant. 
        queueCreateInfo.pQueuePriorities = &queuePriorities;

        /*
        Now we create the logical device. The logical device allows us to interact with the physical
        device.
        */
        VkDeviceCreateInfo deviceCreateInfo = {};

        // Specify any desired device features here. We do not need any for this application, though.
        VkPhysicalDeviceFeatures deviceFeatures = {};
        if (perf.enabled)
            deviceFeatures.pipelineStatisticsQuery = VK_TRUE;

        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.enabledLayerCount = enabledLayers.size();  // need to specify validation layers here as well.
        deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

        VkPhysicalDevicePerformanceQueryFeaturesKHR perfFeatures = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR,
        };

        if (perf.enabled) {
            const char *ext_names[] = { VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME };
            deviceCreateInfo.ppEnabledExtensionNames = ext_names;
            deviceCreateInfo.enabledExtensionCount = 1;

            VkPhysicalDeviceFeatures2 features;
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &perfFeatures;

            vkGetPhysicalDeviceFeatures2(physicalDevice, &features);

            assert(perfFeatures.performanceCounterQueryPools == VK_TRUE);

            deviceCreateInfo.pNext = &perfFeatures;
        }

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

        // Get a handle to the only member of the queue family.
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }

    void createQueries() {
        VkQueryPoolCreateInfo queryPoolCreateInfo;
        queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolCreateInfo.pNext = &perf.performanceQueryCreateInfo;
        queryPoolCreateInfo.flags = 0;
        queryPoolCreateInfo.queryType = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
        queryPoolCreateInfo.queryCount = 1;

        VK_CHECK_RESULT(vkCreateQueryPool(
          device,
          &queryPoolCreateInfo,
          NULL,
          &perf.queryPoolKHR));

        queryPoolCreateInfo.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        queryPoolCreateInfo.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;

        VK_CHECK_RESULT(vkCreateQueryPool(
          device,
          &queryPoolCreateInfo,
          NULL,
          &perf.queryPoolPipeline));
    }

    // find memory type with desired properties.
    uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memoryProperties;

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        /*
        How does this search work?
        See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description. 
        */
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((memoryTypeBits & (1 << i)) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
                return i;
        }
        return -1;
    }

    void createBuffer() {
        /*
        We will now create a buffer. We will render the mandelbrot set into this buffer
        in a computer shade later. 
        */
        
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = bufferSize; // buffer size in bytes. 
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer)); // create buffer.

        /*
        But the buffer doesn't allocate memory for itself, so we must do that manually.
        */
    
        /*
        First, we find the memory requirements for the buffer.
        */
        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
        
        /*
        Now use obtained memory requirements info to allocate the memory for the buffer.
        */
        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size; // specify required memory.
        /*
        There are several types of memory that can be allocated, and we must choose a memory type that:

        1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits). 
        2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
           with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT. 
        Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily 
        visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        this flag.
        */
        allocateInfo.memoryTypeIndex = findMemoryType(
            memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory)); // allocate memory on device.
        
        // Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory. 
        VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, bufferMemory, 0));
    }

    void createDescriptorSetLayout() {
        /*
        Here we specify a descriptor set layout. This allows us to bind our descriptors to 
        resources in the shader. 

        */

        /*
        Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
        0. This binds to 

          layout(std140, binding = 0) buffer buf

        in the compute shader.
        */
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
        descriptorSetLayoutBinding.binding = 0; // binding = 0
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 1; // only a single binding in this descriptor set layout. 
        descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding; 

        // Create the descriptor set layout. 
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    }

    void createDescriptorSet() {
        /*
        So we will allocate a descriptor set here.
        But we need to first create a descriptor pool to do that. 
        */

        /*
        Our descriptor pool can only allocate a single storage buffer.
        */
        VkDescriptorPoolSize descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1; // we only need to allocate one descriptor set from the pool.
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

        // create descriptor pool.
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

        /*
        With the pool allocated, we can now allocate the descriptor set. 
        */
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
        descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
        descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
        descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

        // allocate descriptor set.
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

        /*
        Next, we need to connect our actual storage buffer with the descrptor. 
        We use vkUpdateDescriptorSets() to update the descriptor set.
        */

        // Specify the buffer to bind to the descriptor.
        VkDescriptorBufferInfo descriptorBufferInfo = {};
        descriptorBufferInfo.buffer = buffer;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = bufferSize;

        VkWriteDescriptorSet writeDescriptorSet = {};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSet.dstBinding = 0; // write to the first, and only binding.
        writeDescriptorSet.descriptorCount = 1; // update a single descriptor.
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;

        // perform the update of the descriptor set.
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
    }

    // Read file into array of bytes, and cast to uint32_t*, then return.
    // The data has been padded, so that it fits into an array uint32_t.
    uint32_t* readFile(uint32_t& length, const char* filename) {

        FILE* fp = fopen(filename, "rb");
        if (fp == NULL) {
            printf("Could not find or open file: %s\n", filename);
        }

        // get file size.
        fseek(fp, 0, SEEK_END);
        long filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        long filesizepadded = long(ceil(filesize / 4.0)) * 4;

        // read file contents.
        char *str = new char[filesizepadded];
        fread(str, filesize, sizeof(char), fp);
        fclose(fp);

        // data padding. 
        for (int i = filesize; i < filesizepadded; i++) {
            str[i] = 0;
        }

        length = filesizepadded;
        return (uint32_t *)str;
    }

    void createComputePipeline() {
        /*
        We create a compute pipeline here. 
        */

        /*
        Create a shader module. A shader module basically just encapsulates some shader code.
        */
        uint32_t filelength;
        // the code in comp.spv was created by running the command:
        // glslangValidator.exe -V shader.comp
        uint32_t* code = readFile(filelength, "shaders/comp.spv");
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode = code;
        createInfo.codeSize = filelength;
        
        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
        delete[] code;

        /*
        Now let us actually create the compute pipeline.
        A compute pipeline is very simple compared to a graphics pipeline.
        It only consists of a single stage with a compute shader. 

        So first we specify the compute shader stage, and it's entry point(main).
        */
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = computeShaderModule;
        shaderStageCreateInfo.pName = "main";

        /*
        The pipeline layout allows the pipeline to access descriptor sets. 
        So we just specify the descriptor set layout we created earlier.
        */
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout; 
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;

        /*
        Now, we finally create the compute pipeline. 
        */
        VK_CHECK_RESULT(vkCreateComputePipelines(
            device, VK_NULL_HANDLE,
            1, &pipelineCreateInfo,
            NULL, &pipeline));
    }

    void createCommandPool() {
        /*
        We are getting closer to the end. In order to send commands to the device(GPU),
        we must first record commands into a command buffer.
        To allocate a command buffer, we must first create a command pool. So let us do that.
        */
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        // the queue family of this command pool. All command buffers allocated from this command pool,
        // must be submitted to queues of this family ONLY.
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));
    }

    void allocateCommandBuffers() {
        /*
        Now allocate a command buffer from the command pool.
        */
        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool; // specify the command pool to allocate from.
        // if the command buffer is primary, it can be directly submitted to queues.
        // A secondary buffer has to be called from some primary command buffer, and cannot be directly
        // submitted to a queue. To keep things simple, we use a primary command buffer.
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 2; // allocate 2 command buffers.
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers)); // allocate command buffer.
    }

    void createResetCommandBuffer() {
        /*
        Now we shall start recording commands into the newly allocated command buffer.
        */
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffers[0], &beginInfo)); // start recording commands.

        vkCmdResetQueryPool(commandBuffers[0], perf.queryPoolKHR, 0, 1);
        vkCmdResetQueryPool(commandBuffers[0], perf.queryPoolPipeline, 0, 1);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffers[0])); // end recording commands.
    }

    void createCommandBuffer() {

        /*
        Now we shall start recording commands into the newly allocated command buffer. 
        */
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffers[1], &beginInfo)); // start recording commands.

        if (perf.enabled) {
            vkCmdBeginQuery(commandBuffers[1], perf.queryPoolKHR, 0, 0);
            vkCmdBeginQuery(commandBuffers[1], perf.queryPoolPipeline, 0, 0);
        }

        /*
        We need to bind a pipeline, AND a descriptor set before we dispatch.

        The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        */
        vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

        /*
        Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        The number of workgroups is specified in the arguments.
        If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        */
        vkCmdDispatch(commandBuffers[1],
                (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE_X)),
                (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE_Y)),
                (uint32_t)ceil(DEPTH / float(WORKGROUP_SIZE_Z)));

        if (perf.enabled) {
            vkCmdPipelineBarrier(commandBuffers[1],
              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
              0,
              0, NULL,
              0, NULL,
              0, NULL);

            vkCmdEndQuery(commandBuffers[1], perf.queryPoolKHR, 0);
            vkCmdEndQuery(commandBuffers[1], perf.queryPoolPipeline, 0);
        }

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffers[1])); // end recording commands.
    }

    void runCommandBuffer(VkCommandBuffer *cmdBuf, const void *next) {
        /*
        Now we shall finally submit the recorded command buffer to a queue.
        */

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = next;
        submitInfo.commandBufferCount = 1; // submit a single command buffer
        submitInfo.pCommandBuffers = cmdBuf; // the command buffer to submit.

        /*
          We create a fence.
        */
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));

        /*
        We submit the command buffer on the queue, at the same time giving a fence.
        */
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
        /*
        The command will not have finished executing until the fence is signalled.
        So we wait here.
        We will directly after this read our buffer from the GPU,
        and we will not be sure that the command has finished executing unless we wait for the fence.
        Hence, we use a fence here.
        */
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));

        vkDestroyFence(device, fence, NULL);
    }

    void cleanup() {
        /*
        Clean up all Vulkan Resources. 
        */

        if (enableValidationLayers) {
            // destroy callback.
            auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, NULL);
        }

        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, buffer, NULL);	
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);	
        if (perf.enabled) {
            vkDestroyQueryPool(device, perf.queryPoolKHR, NULL);
            vkDestroyQueryPool(device, perf.queryPoolPipeline, NULL);
        }
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);		
    }
};

int main(int argc, char *argv[]) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s IMG_WIDTH IMG_HEIGHT IMG_DEPTH GROUP_X GROUP_Y GROUP_Z\n", argv[0]);
        exit(1);
    }

    WIDTH = atoi(argv[1]);
    HEIGHT = atoi(argv[2]);
    DEPTH = atoi(argv[3]);
    WORKGROUP_SIZE_X = atoi(argv[4]);
    WORKGROUP_SIZE_Y = atoi(argv[5]);
    WORKGROUP_SIZE_Z = atoi(argv[6]);

    if (WORKGROUP_SIZE_X == 0 || WORKGROUP_SIZE_Y == 0 || WORKGROUP_SIZE_Z == 0||
            WIDTH == 0 || HEIGHT == 0 || DEPTH == 0)
        abort();

    ComputeApplication app;

    try {
        app.run();
    }
    catch (const std::runtime_error& e) {
        printf("%s\n", e.what());
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
