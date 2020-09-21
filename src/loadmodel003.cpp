# include <Siv3D.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <typeinfo> 
#include <thread>
#include <future>

torch::Device device(torch::kCUDA);
torch::Tensor tensor = torch::eye(3).to(device);
torch::data::transforms::Normalize<> normalize_transform({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 });

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


