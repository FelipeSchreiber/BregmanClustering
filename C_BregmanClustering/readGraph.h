#ifndef READGRAPH_H
#define READGRAPH_H
#include <vector>
#include <iostream>
using namespace std;

// Function to add edges
void addEdge(vector<int> adj[], int u, int v);

vector <vector <float> > createGraph(string filename);
#endif