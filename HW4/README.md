## HW 4-1 Policy Gradient

### Game: Pong-v0

This is a practice using Policy Gradient for Pong-v0

To train the model

```
python main.py --train_pg
```

To test the model

```
python main.py --test_pg
```

add improved to train/test PPO model with Policy Gradient

```
python main.py --train_pg_improved
```

```
python main.py --train_pg_improved
```


## HW 4-2 DQN

### Game: Breakout

This is a practice using Deep Q-learning for Breakout


To train the model

```
python main.py --train_XXX
```

To test the model

```
python main.py --test_XXX
```

XXX = dqn, dqn_improved, ddqn, dddqn
for Deep Q-learning, Duel network, Double DQN, Duel and Double DQN


## HW 4-3 Actor Critic

### Game: Breakout & Pong

This is a practice using AC for Breakout and Pong


To train the model

```
python main.py --train_<game>_ac(_improved)
```

<game> = break or pong
() is optional

This practice is implemented on python 3.6.8, keras(2.2.4), h5py(2.9.0), numpy(1.14.5), tensorflow-gpu(1.9.0).
