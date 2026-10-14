# Meeting nodes

- Add explainability or challenge. ← challenge detecting which players are outliers using spectral analysis on directed graph. Note on this: spectral analysis on directed graph is hard. Maybe try on undirected graph. Advice: be open to explaining something, might be easier.

- For comparing use a normalized MSE or similar, arbitrary '40000' doesn't say much

- For comparing models, add std div in plot. e.g. Ridge and MLP are statistically the same.

- Make it clear that MPL is analytical. Maybe not call it attention. Do keep it, just call it different. E.g. normalized edge weights

- More depth in the analysis of the methods, why are things working. Think about what experiments can we do to answer these questions.

- Check out directed graph Fourier transforms for spectral analysis. Could also for example take an explainer not for graphs and translate it to a graph.

- Advice: try to wrap it up in the coming 2 weeks.

- What do you think of the new research question, with outliers? Better than previous for sure, but add little of literature review of outliers etc.

_____________________________________

## After the meeting

- Look into explainers, instead of spectral analysis. He made it pretty clear on directed graph was really hard.

- Take a quick look at directed spectral analysis, to at least have an idea / read a paper about it

- Unified pipeline for model evaluation + preprocessing

- Think about / do more interesting experiments that look at why things perform as they do, and place it into context of other models. E.g. look at how relevant the '0-edges' are, and the alpha values.

- Try to wrap up coding and experiments by the end of week 9, writing deadline is 8th November

_____________________________________

## Tasks

- Unified Pipeline (create interface) <- Ruben + Zhefan

- Experiments MPL <- Karol and Gabriel

- Find some papers and distribute it <- Uros

- Papers of explainers <- Uros

- Paper Gabriel <- Gabriel
