#include <math.h>
#include <stdio.h>
#include <vector>

#include "lodepng.h"
#include "shared.h"

void
save_data(Pixel *data, int width, int height, int depth)
{
    std::vector<unsigned char> image;
    image.reserve(width * height * depth * 4);

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

    for (int i = 0; i < width * height * depth; ++i) {
        fprintf(dataFile, "%u,", i  / (width * height));
        fprintf(dataFile, "%u,", data[i].globalInvocationID.z);

        fprintf(dataFile, "%u,", (i % (width * height)) / width);
        fprintf(dataFile, "%u,", data[i].globalInvocationID.y);

        fprintf(dataFile, "%u,", (i % (width * height)) % width);
        fprintf(dataFile, "%u,", data[i].globalInvocationID.x);

        fprintf(dataFile, "%u,", data[i].workGroupID.z);
        fprintf(dataFile, "%u,", data[i].numWorkGroups.z);

        fprintf(dataFile, "%u,", data[i].workGroupID.y);
        fprintf(dataFile, "%u,", data[i].numWorkGroups.y);

        fprintf(dataFile, "%u,", data[i].workGroupID.x);
        fprintf(dataFile, "%u,", data[i].numWorkGroups.x);

        fprintf(dataFile, "%u,", data[i].localInvocationID.z);
        fprintf(dataFile, "%u,", data[i].workGroupSize.z);

        fprintf(dataFile, "%u,", data[i].localInvocationID.y);
        fprintf(dataFile, "%u,", data[i].workGroupSize.y);

        fprintf(dataFile, "%u,", data[i].localInvocationID.x);
        fprintf(dataFile, "%u,", data[i].workGroupSize.x);

        fprintf(dataFile, "%u,", data[i].localInvocationIndex.x);

        fprintf(dataFile, "%u,", data[i].subgroup.x); // SGID
        fprintf(dataFile, "%u,", data[i].subgroup.w); // NumSG

        fprintf(dataFile, "%u,", data[i].subgroup.y); // SGIID
        fprintf(dataFile, "%u,", data[i].subgroup.z); // SGS

        fprintf(dataFile, "%f,", data[i].r);
        fprintf(dataFile, "%u,", (unsigned char)(255.0f * (data[i].r)));
        fprintf(dataFile, "%f,", data[i].g);
        fprintf(dataFile, "%u,", (unsigned char)(255.0f * (data[i].g)));
        fprintf(dataFile, "%f,", data[i].b);
        fprintf(dataFile, "%u,", (unsigned char)(255.0f * (data[i].b)));
        fprintf(dataFile, "%f,", data[i].a);
        fprintf(dataFile, "%u", (unsigned char)(255.0f * (data[i].a)));

        fprintf(dataFile, "\n");

        image.push_back((unsigned char)(255.0f * (data[i].r)));
        image.push_back((unsigned char)(255.0f * (data[i].g)));
        image.push_back((unsigned char)(255.0f * (data[i].b)));
        image.push_back((unsigned char)(255.0f * (data[i].a)));
    }

    fclose(dataFile);

    unsigned error;

    const bool grid = true;
    if (grid) {
        /* this could be done on the GPU if the purpose of this test
         * would be to this as quickly as possible */
        std::vector<unsigned char> image2;
        image2.reserve(width * height * depth * 4);
        int columns = (int)ceil(sqrt(depth));
        while (depth % columns != 0)
            columns++;

        for (int r = 0; r < depth / columns; ++r) {
            for (int h = 0; h < height; ++h) {
                for (int c = 0; c < columns; ++c) {
                    std::vector<unsigned char>::iterator it =
                            image.begin() + 4 * ((r * columns + c) * width * height + h * width);
                    image2.insert(image2.end(), it, it + width * 4);
                }
            }
        }
        error = lodepng::encode("result.png", image2, width * columns, height * depth / columns);
    } else {
        error = lodepng::encode("result.png", image, width, height * depth);
    }

    if (error)
        printf("encoder error %d: %s", error, lodepng_error_text(error));
}
