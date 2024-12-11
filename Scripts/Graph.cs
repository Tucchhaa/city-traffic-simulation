using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Serialization;

[System.Serializable]
public class GraphNode
{
    public GameObject from;
    public List<GameObject> to;

    public GraphNode(GameObject from, List<GameObject> to)
    {
        this.from = from;
        this.to = to;
    }
}

public class Graph : MonoBehaviour
{
    public List<GraphNode> graph = new ();

    private const float Inf = 1e9f;

    private List<List<float>> _matrix;
    private List<List<int>> _parent;
    private readonly Dictionary<int, GameObject> _gameObjects = new();
    private readonly Dictionary<int, int> _indexToInstanceID = new();
    private readonly Dictionary<int, int> _instanceIDToIndex = new();
    
    private void Start()
    {
        ConstructGraph();
        CalculateShortestPaths();
    }
    
    public List<GameObject> GetShortestPath(GameObject a, GameObject b)
    {
        var indA = _instanceIDToIndex[a.GetInstanceID()];
        var indB = _instanceIDToIndex[b.GetInstanceID()];
    
        if (_parent[indA][indB] == -1)
            return new List<GameObject>();
    
        var path = new List<GameObject> { GetNode(indA) };

        while (indA != indB)
        {
            indA = _parent[indA][indB];
            path.Add(GetNode(indA));
        }
    
        return path;

        GameObject GetNode(int nodeIndex)
        {
            var instanceID = _indexToInstanceID[nodeIndex];
            var node = _gameObjects[instanceID];
    
            return node;
        }
    }
    
    private void ConstructGraph()
    {
        foreach (var node in graph)
        {
            Add(node.from);

            foreach (var to in node.to)
            {
                Add(to);
            }
        }
    
        var N = _gameObjects.Count();
    
        _matrix = new List<List<float>>(N);
        _parent = new List<List<int>>(N);
    
        for (var i = 0; i < N; i++)
        {
            _matrix.Add(Enumerable.Repeat(Inf, N).ToList());
            _parent.Add(Enumerable.Repeat(-1, N).ToList());
            
            _matrix[i][i]  = 0;
        }

        foreach (var node in graph)
        {
            var from = node.from;
            var fromInd = _instanceIDToIndex[node.from.GetInstanceID()];

            foreach (var to in node.to)
            {
                var toInd = _instanceIDToIndex[to.GetInstanceID()];
                
                var distance = Vector3.Distance(from.transform.position, to.transform.position);
                
                _matrix[fromInd][toInd] = distance;
                _parent[fromInd][toInd] = toInd;
            }
        }

        return;

        void Add(GameObject gameObject)
        {
            var id = gameObject.GetInstanceID();

            if (_gameObjects.ContainsKey(id)) return;
            
            _indexToInstanceID[_gameObjects.Count()] = id;
            _instanceIDToIndex[id] = _gameObjects.Count();
            _gameObjects[id] = gameObject;
        }
    }
    
    private void CalculateShortestPaths()
    {
        var N = _matrix.Count;
    
        for (var k = 0; k < N; k++)
        {
            for (var i = 0; i < N; i++)
            {
                for (var j = 0; j < N; j++)
                {
                    if (_matrix[i][j] > _matrix[i][k] + _matrix[k][j])
                    {
                        _matrix[i][j]  = _matrix[i][k] + _matrix[k][j];
                        _parent[i][j] = _parent[i][k];
                    }
                }
            }
        }
    }
}
