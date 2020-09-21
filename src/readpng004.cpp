//#include <torch/torch.h>
#include <Siv3D.hpp>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <iostream>
#include <chrono>
#include <iostream>
#include <typeinfo>
#include <thread>
#include <future>
#include "../include/utils/vision_utils.hpp"
using namespace std;
using namespace std::chrono;

torch::Device device(torch::kCUDA);
torch::Tensor tensor = torch::eye(3).to(device);

void Main()
{
    Window::SetTitle(U"TorchSiv3D C++");
    const Texture icn0(Emoji(U"✡"));
    icn0.draw(0, 0);
    Scene::SetBackground(Color(87, 83, 95));


    auto VU = VisionUtils();
    torch::Device device(torch::kCUDA);
    //torch::Device device(torch::kCPU);
    // Input PNG-image
    png::image<png::rgb_pixel> imageI("siv3d-kun.png");
    // Convert png::image into torch::Tensor
    torch::Tensor tensor = VU.pngToTorch(imageI, device); //Note: we are allocating on the GPU
    
    // Convert torch::Tensor into png::image
    png::image<png::rgb_pixel> imageO = VU.torchToPng(tensor.detach().cpu()); // if we do not move teh tensor to the CPU= seg fault 
    // Input PNG-image
    imageO.write("siv3d-kun-output001.png");    

    // 🐈 の絵文字からテクスチャを作成
    const Texture texture(Emoji(U"🍉"));

    //read the output from PyTorch! 
    const Texture textureSiv3DKun(U"siv3d-kun-output001.png", TextureDesc::Mipped);

	while (System::Update())
	{   
        const double scale = 0. + Periodic::Triangle0_1(10s) * 0.6;

        Print(U"C:", tensor.size(0), U"H:", tensor.size(1), U"W:", tensor.size(2));
        
        // テクスチャを画面中心から描画
        texture.draw(Scene::Center());

        textureSiv3DKun.scaled(scale).drawAt(400, 300);;
	}
}