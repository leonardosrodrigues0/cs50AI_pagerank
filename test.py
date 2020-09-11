from pagerank import transition_model

print(transition_model(
    {"1.html": {}, "2.html": {"3.html"}, "3.html": {"2.html"}, "hey": {"hello"}},
    "1.html",
    0.85
))