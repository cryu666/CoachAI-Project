# Track 2: Forecasting Future Turn-Based Strokes in Badminton Rallies

## :badminton: Task Introduction
The goal of this track is to **forecast future strokes including shot types and locations given the past stroke sequences**, namely stroke forecasting. For each singles rally, given the observed 4 strokes with type-area pairs and two players, the goal is to predict the future strokes including shot types and area coordinates for the next n steps. n is various based on the length of the rally.

## :badminton:	Data Overview
* Input: landing_x, landing_y, shot type of past 4 strokes 
* Output: landing_x, landing_y, shot type of future strokes 


## :badminton:	Evaluation Metrics

$$ 
\begin{flalign}
&Score = min(l_1, l_2, ..., l_6)&
\end{flalign}
$$

$$ 
\begin{flalign}
l_i = AVG(CE + MAE)&
= \frac{ \sum_{r=1}^{|R|} \sum_{n=\tau+1}^{|r|} [S_n log \hat{S_n} + (|x_n-\hat{x_n}|+|y_n-\hat{y_n}|)]} {|R|\cdot(|r|-\tau)} &
\end{flalign}
$$


## :badminton:	Main approach: ShuttleNet
### Overview
ShuttleNet is a deep learning model that can be applied to stroke forecasting in badminton. It typically consists of two encoder-decoder extractors for modeling rally progress and retrieving player styles from turn-based sequence and a fusion network to take into account the dependencies between rally progress and player styles at each stroke.
Please refer to the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20341) for more details.
Here we adapt ShuttleNet to our newly collected dataset as the official baseline in the CoachAI Badminton Challenge.
All hyper-parameters are set as the same in the paper.

### Code Usage
#### Train a model
```=bash
./script.sh
```

#### Generate predictions
```=bash
python generator.py {model_path}
```

#### Run evaluation metrics
- Both ground truth and prediction files are default in the `data` folder
```=bash
python evaluation.py
```
