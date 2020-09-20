# include <Siv3D.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <typeinfo> 
#include <thread>
#include <future>
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../include/stb_image_resize/stb_image_resize.h"

torch::Device device(torch::kCUDA);
torch::Tensor tensor = torch::eye(3).to(device);
torch::data::transforms::Normalize<> normalize_transform({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 });

//Adapted from https://github.com/prabhuomkar/pytorch-cpp/tree/master/utils/image_io
// Loads a tensor from an image file
torch::Tensor load_image(const std::string& file_path,
    torch::IntArrayRef shape, std::function<torch::Tensor(torch::Tensor)> transform) {
    if (!shape.empty() && shape.size() != 1 && shape.size() != 2) {
        throw std::invalid_argument("Shape must be empty or contain exactly one or two elements.");
    }

    int width = 0;
    int height = 0;
    int depth = 0;

    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image_raw(stbi_load(file_path.c_str(),
        &width, &height, &depth, 0), &stbi_image_free);

    if (!image_raw) {
        throw std::runtime_error("Unable to load image file " + file_path + ".");
    }

    if (shape.empty()) {
        return transform(torch::from_blob(image_raw.get(),
            { height, width, depth }, torch::kUInt8).clone().to(torch::kFloat32).permute({ 2, 0, 1 }).div_(255));
    }

    int new_width = 0;
    int new_height = 0;

    if (shape.size() == 1) {
        double scale = static_cast<double>(shape[0]) / std::max(width, height);
        new_width = width * scale;
        new_height = height * scale;
    }
    else {
        new_width = shape[1];
        new_height = shape[0];
    }

    if (new_width < 0 || new_height < 0) {
        throw std::invalid_argument("Invalid shape.");
    }

    size_t buffer_size = new_width * new_height * depth;

    std::vector<unsigned char> image_resized_buffer(buffer_size);

    stbir_resize_uint8(image_raw.get(), width, height, 0,
        image_resized_buffer.data(), new_width, new_height, 0, depth);

    return transform(torch::from_blob(image_resized_buffer.data(),
        { new_height, new_width, depth }, torch::kUInt8).clone().to(torch::kFloat32).permute({ 2, 0, 1 }).div_(255));
}


void tensorDIMS(const torch::Tensor& tensor) {
    auto t0 = tensor.size(0);
    auto s = tensor.sizes();
    Print (tensor.size(0), U",", tensor.size(1), U",", tensor.size(2));
    //Print(tensor.size(0));
}

void Main()
{
	Window::SetTitle(U"TorchSiv3D C++");
	const Texture icn0(Emoji(U"✡"));
	icn0.draw(0, 0);				
	Scene::SetBackground(Color(90, 81, 95));    
    
    const std::string modelName = "erfnet_fs.pt";
    const std::string content_image_path = "windmill.png";

    auto module = torch::jit::load(modelName, device);
    //module->to(at::kCUDA);
    if (!std::ifstream(modelName)) {

        Print  (U"ERROR: Could not open the required trained PyTorch module file from path");
    }
    else {
        Print(U"Loaded required trained PyTorch module module file from path");
    }
    assert(module != nullptr);
    const int64_t max_image_size = 256;
    auto content = load_image(content_image_path, max_image_size, normalize_transform).unsqueeze_(0);
    
    tensorDIMS(content);
    //auto x = (content).data()[0]; // Move it to the CPU
    //Print(x); //Use it from Siv3D
	while (System::Update())
	{	
        

	}
}


