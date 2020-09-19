<h1 align="center">  
  <img src="TORCHLOGO.png"></a>
</h1>

<h4 align="center">An integration of the Siv3D C++ framework with the Libtorch C++ Deep Learning Library</h4>
      
<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •  
  <a href="#features">Features</a> •  
  <a href="#author">Author</a> •  
  <a href="#license">License</a>
</p>

---

## About

<table>
<tr>
<td>
  
**Siv3DTorch++** is an **integration** of the well-known Japanese **_OpenSiv3D_** (https://github.com/Siv3D/OpenSiv3D) creative coding library (https://siv3d.github.io/) and my favourite Deep Learning Library Libtorch: the **_PyTorch_** C++ frontend. 
The integration allows one to:
 
![Siv3DTorch++ Code](https://github.com/QuantScientist/Siv3DTorch/blob/master/vc19001.png?raw=true)
<p align="right">
<sub>(Preview)</sub>
</p>

</td>
</tr>
</table>

## A simple example 
The folowing example allocates a PyTorch style random tensor on the GPU ( a CPU is also supported of course), applies the sigmoid to it, then detaches the tensor from 
the GPU and uses the result to display on a Siv3D window.
 
```cpp
torch::Tensor sigmoid001(const torch::Tensor & x ){
    torch::Tensor sig = 1.0 / (1.0 + torch::exp(( -x)));
    return sig;
}
```

![Siv3DTorch++ Code](https://github.com/QuantScientist/Siv3DTorch/blob/master/simple001.gif?raw=true)

Full source code:

```cpp
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
	Scene::SetBackground(Color(87, 83, 95));					
					
	while (System::Update())
	{	
		for (auto i : Range(1, 20))
		{
			//ClearPrint();			
			//torch::Tensor t0 = torch::tensor((i)).to(device);
			torch::Tensor t0 = torch::rand(1).to(device); // Allocate a tensor on the GPU
			t0 = sigmoid001(t0);
			//Print (typeid(t0).name());		
			auto x = (t0).data().detach().item().toFloat(); // Move it to teh CPU
			Print(x); //Use it from Siv3D			
			Circle(300 * (x), 300*x, 50*x).draw((ColorF(0.5 *x, 0.9*x, 0.3*x)));
		}		
	}
}
```

## Installation

##### Downloading and installing steps:
* **[Download](https://github.com/QuantScientist/Siv3DTorch/archive/master.zip)** the latest version of the config.
* **Go** to the following path: `\...\Steam\userdata\<Your_SteamID3>\730\local\`
  * See below **[how to find your SteamID3](https://github.com/ArmynC/ArminC-AutoExec#how-to-find-your-steamid3)**.
* Place the **cfg** folder (from .zip) inside the **local** folder (from the path).
  * Replace all files if it asks.
    * To use the **Video Settings**, rename `video_optional.txt` to `video.txt` and set it to `Read-only`.
* **[OPTIONAL]** Set the **[launch options](https://github.com/QuantScientist/Siv3DTorch/wiki/Launch-Options)**.
  * **Right-click** on the **game title** under the _Library_ in Steam and select **Properties**.
  * Under the **General tab** click the **Set launch options...** button.
  * **Enter** the **launch options** you wish to apply (_be sure to separate each code with space_) and click **OK**.
  * **Close** the _Properties_ window and **launch the game**
* **Launch** the game and **type** in the _console_ the following command: `exec autoexec.cfg`

## Features

|                            | 🔰 Siv3DTorch++ VC 19  | ◾ CMake |
| -------------------------- | :----------------: | :-------------: |
| PyTorch CPU tensors        |         ✔️         |        ❌        |
| PyTorch GPU tensors        |         ✔️         |        ❌        |
| Libtorch 1.6               |         ✔️         |        ❌        |


## Backtesting Signal Accuracy
During the testing period, the model signals to buy or sell based on its prediction for price
movement the following day. By putting your trading algorithm aside and testing for signal accuracy
alone, you can rapidly build and test more reliable models.


## Contributing

Feel free to report issues during build or execution. We also welcome suggestions to improve the performance of this application.

## Author
Shlomo Kashani, Head of AI at DeepOncology AI, Kaggle Expert, Author of the book _Deep Learning Interviews_: entropy@interviews.ai 

## Citation

If you find the code or trained models useful, please consider citing:

```
@misc{Siv3DTorch++,
  author={Kashani, Shlomo},
  title={Siv3DTorch++2020},
  howpublished={\url{https://github.com/QuantScientist/Siv3DTorch/}},
  year={2020}
}
```

## License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-orange.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

- Copyright © [Shlomo](https://github.com/QuantScientist/).
