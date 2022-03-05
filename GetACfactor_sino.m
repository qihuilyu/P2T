function ci = GetACfactor_sino(G, mumap)

li = G * mumap;
ci = exp(-li);
