/*
Universidade Federal do Rio de Janeiro
Author: Felipe Schreiber Fernandes

$Author$
$Date$
$Log$

*/

#include <iostream>
#include "readGraph.h"
using namespace std;

// Function to add edges
void addEdge(vector<int> adj[], int u, int v)
{
    adj[u].push_back(v);
}

