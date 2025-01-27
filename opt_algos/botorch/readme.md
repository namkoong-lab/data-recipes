I'm confused why model size is the same as fidelity 

In our model, there's: 
x = [model_size, data proportions]
fidelity = train_steps/how much compute to spend
y = target

why is it that thomson is saying:
x = [ data proportions]
fidelity =[ trains_steps/how much compute to spend, model size]
y = target

is it because fidelity has a monotonic element to it, i.e. we start small and go bigger. so we want to treat
model size as a monotonically growing function as well. i.e. we start small and go big. 


So main goal is we want to substitute our simulator in the botorch code and show it works.



Model size is not technically monotonic, our bayes opt algo might suddenly decide to go from big size back to small size just to test things out.
Only thing is we want to implement earlier constraints to make sure it doesn't start at 1B. 


ah nvm, im wrong, model size is a fidelity.
our bayes opt goal at the end is to output the data_mixtures vector x that is best for the largest model size.
by the above dfn, then both training steps and model sizes are fidelities, i.e. lower clarity runs that should inform the final run


TODO:

1. make sure we can have clear metric and to show how EI and KG differ over time. 
2. Make it output data that we can plot over later
3. Also make some plots
4. Add model size as a fidelity that we can 
2. turn it into a .py file we can run in the background