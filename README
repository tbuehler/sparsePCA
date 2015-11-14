SPARSE PCA VIA NONLINEAR INVERSE POWER METHOD


This archive contains a Matlab implementation of Sparse PCA
using the inverse power method for nonlinear eigenproblems as 
described in the paper

M. Hein and T. Buehler
An Inverse Power Method for Nonlinear Eigenproblems with Applications 
in 1-Spectral Clustering and Sparse PCA
In Advances in Neural Information Processing Systems 23 (NIPS 2010).
(Extended version available online at http://arxiv.org/abs/1012.0774)


Current version: V1.0



SHORT DOCUMENTATION

Usage:
[cards,vars,Z]= sparsePCA(X,card);
[cards,vars,Z]= sparsePCA(X,card_min,card_max);
[cards,vars,Z]= sparsePCA(X,card_min,card_max,numRuns);
[cards,vars,Z]= sparsePCA(X,card_min,card_max,numRuns,verbosity);

X : data matrix (num x dim)
card : desired number of non-sparse components of output (cardinality)
card_min,card_max : computes all vectors with cardinality values in 
     intervall [card_min,card_max] (default: card_min=card_max)
numRuns : number of runs of inverse power method with random 
     initialization (default: 10)
verbosity [0-2]: determines how much information is displayed (default: 1)

cards : the cardinalities (number of nonzero components) of the 
	returned vectors 
vars : the corresponding vectors
Z : the sparse principal components

	
	
LICENSE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

If you use this code for your publication, please include a reference 
to the paper "An inverse power method for nonlinear eigenproblems with 
applications in 1-spectral clustering and sparse PCA".
 
 
CONTACT

(C)2010-2011 Thomas Buehler and Matthias Hein
tb,hein@cs.uni-saarland.de
Machine Learning Group, Saarland University, Germany
http://www.ml.uni-saarland.de
