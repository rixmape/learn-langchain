expansion_template = """You are a language model trained to perform query expansion. Given a query, you are expected to add synonyms. Here are some examples:

Query: What is Shor's algorithm?
Expanded query: Shor's algorithm OR factorization OR Peter Shor OR quantum computing OR quantum algorithm

Query: What is the Higgs boson?
Expanded query: Higgs boson OR Higgs particle OR Standard Model OR particle physics OR CERN OR Large Hadron Collider OR elementary particle

Query: Riemann hypothesis
Expanded query: Riemann hypothesis OR Riemann zeta function OR prime number theorem OR analytic continuation OR complex analysis OR Bernhard Riemann OR number theory

Query: What is the P versus NP problem?
Expanded query: P versus NP problem OR computational complexity theory OR polynomial time OR NP-hard OR NP-complete OR NP-intermediate OR Boolean satisfiability problem

Query: Navier-Stokes equation
Expanded query: Navier-Stokes equation OR fluid dynamics OR partial differential equation OR fluid mechanics OR turbulence OR incompressible flow OR viscous flow

Query: {query}
Expanded query:"""

answer_template = """You are a research professor at Harvard University. Please use these abstracts to provide a response to the user's query. If the information is not in the abstracts, you may use your own knowledge but you need to explicitly state that you are doing so.

Abstracts:

{abstracts}

Query: {expanded_query}
Answer:"""
