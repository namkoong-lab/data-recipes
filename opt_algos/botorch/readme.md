Model size is not technically monotonic, our bayes opt algo might suddenly decide to go from big size back to small size just to test things out.
Only thing is we want to implement earlier constraints to make sure it doesn't start at 1B. 


ah nvm, im wrong, model size is a fidelity.
our bayes opt goal at the end is to output the data_mixtures vector x that is best for the largest model size.
by the above dfn, then both training steps and model sizes are fidelities, i.e. lower clarity runs that should inform the final run


TODO DONE:

1. make sure we can have clear metric and to show how EI and KG differ over time. 
2. Make it output data that we can plot over later
3. Also make some plots
2. turn it into a .py file we can run in the background


TODO:
1. Add Wandb
4. Add model size as a fidelity that we can optimze over
3. Make EI into multi-fidelity



Idea 1:
GP : 7 dim : [w1 ... w5, model_size, step]
fidelities : [model_size, step]


Thomsons:

One step lookahead:
Fix the model size, and then optimize over the step.

def Compute KG(vector of posterior mean across steps, flops(m), division_coefficient): R196
    """
    vector of posterior mean across steps: R196
    flops(m): R196
    division_coefficient: R196

    division_coefficient : scalar to let us control the relative importance of the IG and flops(m)
    """





for i in iterations:
    all highest IG = []
    For m in model_sizes:
        vector of posterior mean across steps = Compute GP(m): R196
        vector of IG across steps = Compute KG(vector of posterior mean across steps, flops(m)): R196
        highest IG = argmax(vector of IG across steps)
        all highest IG.append(highest IG)

    chosen_step = argmax(all highest IG)
    observe datamodel(chosen_step)




Random Idea: We can normalize the KG by the flops(m). 


--------------------------------

Can't we just compute EI over all points, then just divide them by the flops(m)?
So my acquisition function is just EI(x)/flops(m).

--------------------------------


ask about multi fidelity EI. 





Idea 2:
GP : 6 dim : [w1 ... w5, flops]
flops = cost(model_size, step)





--------------------------------

ok so EI not having MF makes sense,, because it is taking the maximal EI, naturally that's the one with the highest model size and max training steps.

but KG is computing the information gain, not the EI. So it'll do better with MF

GOAL: Modify their training sample code to have the 7 dims. and compare. 

