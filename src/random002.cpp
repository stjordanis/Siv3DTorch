# include <Siv3D.hpp>
#include <torch/script.h>
#include <vector>
#include <typeinfo> 

torch::Tensor sigmoid001(const torch::Tensor& x) {	
	torch::Tensor sig = 1.0 / (1.0 + torch::exp((-x)));
	return sig;
}

torch::Device device(torch::kCUDA);
torch::Tensor tensor = torch::eye(3).to(device);

void Main()
{
	Window::SetTitle(U"TorchSiv3D C++");
	
	const Texture icn0(Emoji(U"✡"));
	icn0.draw(0, 0);				
	Scene::SetBackground(Color(90, 81, 95));				

	//const Texture texture(Image(L"Example/Windmill.png").scale(0.3));	
					
	while (System::Update())
	{	
		//constexpr RectF shape(100, 100, 400, 300);
		constexpr Circle shape(300, 200, 150);

		Array<Vec2> points;

		shape.draw(Palette::Black);
		
		for (size_t i = 0; i < 100; ++i)
		{
			points << RandomVec2(shape);
		}

		for (const auto& point : points)
		{
			Circle(point, 3* torch::rand(1).to(device).data().detach().item().toFloat()).draw(Palette::Orange);
		}

		for (auto i : Range(1, 30))
		{
			//ClearPrint();			
			//torch::Tensor t0 = torch::tensor((i)).to(device);
			torch::Tensor t0 = torch::rand(1).to(device); // Allocate a tensor on the GPU
			t0 = sigmoid001(t0);
			//Print (typeid(t0).name());		
			auto x = (t0).data().detach().item().toFloat(); // Move it to the CPU
			Print(x); //Use it from Siv3D			
			Circle(300 * (x), 300*x, 100*x).draw(
				(ColorF(
				0.5 * torch::rand(1).to(device).data().detach().item().toFloat(),
				0.9*x, 
				0.3*x)));
		}	

		int sx = Random(Window::ClientWidth()) / 2;
		int sy = Random(Window::ClientWidth()) / 2;
		int dx = 200 + Random(20);
		int dy = 10 + Random(10);
		RectF rect = RectF(sx, sy, dx, dy);
		rect.draw();
	}
}


