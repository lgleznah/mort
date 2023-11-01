#ifndef IMG_LOADER_H
#define IMG_LOADER_H

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
#pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "external/stb_image.h"

#include <cstdlib>
#include <iostream>

class img_loader {
public:
    img_loader() : data(nullptr) {}

    img_loader(const char* image_filename) {
        // Loads image data from the specified file. If the RTW_IMAGES environment variable is
        // defined, looks only in that directory for the image file. If the image was not found,
        // searches for the specified image file first from the current directory, then in the
        // images/ subdirectory, then the _parent's_ images/ subdirectory, and then _that_
        // parent, on so on, for six levels up. If the image was not loaded successfully,
        // width() and height() will return 0.

        auto filename = std::string(image_filename);

        // Hunt for the image file in some likely locations.
        if (load(filename)) return;

        std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
    }

    ~img_loader() { STBI_FREE(data); }

    bool load(const std::string filename) {
        // Loads image data from the given file name. Returns true if the load succeeded.
        auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
        data = stbi_load(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
        bytes_per_scanline = image_width * bytes_per_pixel;
        return data != nullptr;
    }

    int width()  const { return (data == nullptr) ? 0 : image_width; }
    int height() const { return (data == nullptr) ? 0 : image_height; }
    int bytes_scanline() const { return (data == nullptr) ? 0 : bytes_per_scanline; }
    int bytes_pixel() const { return (data == nullptr) ? 0 : bytes_per_pixel; }
    unsigned char* raw_data() const { return (data == nullptr) ? nullptr : data; }

    const unsigned char* pixel_data(int x, int y) const {
        // Return the address of the three bytes of the pixel at x,y (or magenta if no data).
        static unsigned char magenta[] = { 255, 0, 255 };
        if (data == nullptr) return magenta;

        x = clamp(x, 0, image_width);
        y = clamp(y, 0, image_height);

        return data + y * bytes_per_scanline + x * bytes_per_pixel;
    }

private:
    const int bytes_per_pixel = 3;
    unsigned char* data;
    int image_width, image_height;
    int bytes_per_scanline;

    static int clamp(int x, int low, int high) {
        // Return the value clamped to the range [low, high).
        if (x < low) return low;
        if (x < high) return x;
        return high - 1;
    }
};

// Restore MSVC compiler warnings
#ifdef _MSC_VER
#pragma warning (pop)
#endif

#endif