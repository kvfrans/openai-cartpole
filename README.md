# Random search, hill climbing, policy gradient for CartPole

Simple reinforcement learning algorithms implemented for CartPole on OpenAI gym.

This code goes along with [my post about learning CartPole](http://kvfrans.com/simple-algoritms-for-solving-cartpole/), which is inspired by [an OpenAI request for research](https://openai.com/requests-for-research/#cartpole).

##Algorithms implemented

**Random Search**: Keep trying random weights between [-1,1] and greedily keep the best set.

**Hill climbing**: Start from a random initialization, add a little noise evey iteration and keep the new set if it improved.

**Policy gradient** Use a softmax policy and compute a value function using discounted Monte-Carlo. Update the policy to favor action-state pairs that return a higher total reward than the average total reward of that state. Read [my post about learning CartPole](http://kvfrans.com/simple-algoritms-for-solving-cartpole/) for a better explanation of this.
