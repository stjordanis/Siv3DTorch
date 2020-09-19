<h1 align="center">  
  <img src="TORCHLOGO.png"></a>
</h1>

<h4 align="center">An integration of the Siv3D C++ framework with the Libtorch C++ Deep Learning Library</h4>

<p align="center">
    <a href="https://github.com/QuantScientist/Siv3DTorch/commits/master">
    <img src="https://img.shields.io/github/last-commit/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit">
    <a href="https://github.com/QuantScientist/Siv3DTorch/issues">
    <img src="https://img.shields.io/github/issues-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues">
    <a href="https://github.com/QuantScientist/Siv3DTorch/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests">
    <a href="https://twitter.com/intent/tweet?text=Try this CS:GO AutoExec:&url=https%3A%2F%2Fgithub.com%2FArmynC%2FArminC-AutoExec">
    <img src="https://img.shields.io/twitter/url/https/github.com/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=twitter"
         alt="GitHub tweet">
</p>
      
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
  
**Siv3DTorch++** is a **high-quality** _config_ for _Counter-Strike: Global Offensive_ that aims to **improve the gameplay** and its **template is perfect**, enabling you to **customize** the game **settings** to your liking.

It comes **filled** with **optimizations** that make use of the **all network capacity** and **game advantages**, overall **improving the gameplay** for a wide variety of _computers and players_.

**Each and one** of the **commands** are **finely-tuned**, _enabling all game's capacity_ (compared to default settings), **helping you** through the matchmaking.

![Siv3DTorch++ Code](https://github.com/QuantScientist/Siv3DTorch/blob/master/TORCHLOGO.png?raw=true)
<p align="right">
<sub>(Preview)</sub>
</p>

</td>
</tr>
</table>

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

##### How to find your SteamID3:

* **Go** to [SteamID](https://steamid.io/).
* In the _input_ box, **enter** your **profile** name/id and press ***lookup**.
* All the SteamIDs versions will be shown. You need **SteamID3**.
  * The format: `[X:Y:ZZZZZZZZ]` - where your *install path id* is the whole `Z` code.

## Updating

When a **new version** is out, you have **two methods** to _update_:

##### 1. You have edited the config based on your preference:
* Check the new [commits](https://github.com/QuantScientist/Siv3DTorch/commits/master) and **update** the config **manually** by relying on the _commits_.

##### 2. You haven't edited the config (or at least not so much):
* **Delete everything** (or **replace the files** when it asks).
* **Redo** the [installation](https://github.com/ArmynC/ArminC-AutoExec#installation) steps.
* _After setup_, **change your preference** settings back (if it is the case).

This _config_ is **updated** (at a random time), so make sure you **come back** here to **check** for **updates**.

## Features

|                            | 🔰 Siv3DTorch++  | ◾ Other Configs |
| -------------------------- | :----------------: | :-------------: |
| Optimized values           |         ✔️         |        ❌        |
| Useful scripts             |         ✔️         |        ❌        |
| Documented commands        |         ✔️         |        ❌        |
| Enabled in-game advantages |         ✔️         |        ❌        |
| No misconcepted commands   |         ✔️         |        ❌        |
| Professional info sources  |         ✔️         |        ❌        |
| Clean sheet/template       |         ✔️         |        ❌        |
| Easy to customize          |         ✔️         |        ❌        |
| Categorized by functions   |         ✔️         |        ❌        |
| New commands/values        |         ✔️         |        ❌        |
| No old command leftovers   |         ✔️         |        ❌        |


## Backtesting Signal Accuracy
During the testing period, the model signals to buy or sell based on its prediction for price
movement the following day. By putting your trading algorithm aside and testing for signal accuracy
alone, you can rapidly build and test more reliable models.

```python
from clairvoyant.engine import Backtest
import pandas as pd

features  = ["EMA", "SSO"]   # Financial indicators of choice
trainStart = 0               # Start of training period
trainEnd   = 700             # End of training period
testStart  = 701             # Start of testing period
testEnd    = 1000            # End of testing period
buyThreshold  = 0.65         # Confidence threshold for predicting buy (default = 0.65) 
sellThreshold = 0.65         # Confidence threshold for predicting sell (default = 0.65)
continuedTraining = False    # Continue training during testing period? (default = false)
```

## Contributing

Feel free to report issues during build or execution. We also welcome suggestions to improve the performance of this application.

## Author
Shlomo Kashani, Head of AI at DeepOncology AI, 
Kaggle Expert, Founder of Tel-Aviv Deep Learning Bootcamp: shlomo@deeponcology.ai

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
