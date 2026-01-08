#Bib: From bib Pyformlang

#regex--Convert-->dfa
#dfa--Minimize-->minimalDfa

from automata_toolkit.regex_to_nfa import regex_to_nfa
from automata_toolkit.dfa_to_efficient_dfa import dfa_to_efficient_dfa
from automata_toolkit.nfa_to_dfa import nfa_to_dfa
from automata_toolkit.visual_utils import draw_nfa
from automata_toolkit.visual_utils import draw_dfa




regx = "ab*"

dfa_test = dfa_to_efficient_dfa()
nfa = regex_to_nfa(regx)

dfa = nfa_to_dfa(nfa)
##draw_nfa(nfa, "nfa_from_regex.png")
##draw_dfa(dfa, "dfa_from_regex.png")

mindfa = dfa_to_efficient_dfa(dfa)
print("before: ", mindfa)

a = list(enumerate(mindfa['states']))
print("after: ", a)

##draw_dfa(mindfa, "min_dfa_from_regex.png")

"""

#lange bis es berechnet wurde
min_dfa = dfa_to_efficient_dfa(nfa_to_dfa(regex_to_nfa("abb*c*")))
"""





