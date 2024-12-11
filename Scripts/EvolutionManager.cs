using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine.Serialization;

internal class Genotype
{
    public readonly List<double> Weights;
    public float Fitness;

    public Genotype(List<double> weights, float fitness)
    {
        Weights = weights;
        Fitness = fitness;
    }
}

internal class Vehicle
{
    public Genotype Genotype;
    public GameObject GameObject;
    public int ID;
    public VehicleController Controller;

    public List<GameObject> Path;
    private int _pathIndex = 1;
    
    private const float _distanceSqrToAchieve = 225.0f;

    public Vehicle(Genotype genotype, GameObject gameObject, List<GameObject> path)
    {
        Genotype = genotype;
        GameObject = gameObject;
        ID = gameObject.GetInstanceID();
        Controller = gameObject.GetComponent<VehicleController>();

        Path = path;
    }

    public void Reset()
    {
        _pathIndex = 1;
    }
    
    public float GetFitness()
    {
        float progressScore;

        float total = (Path[_pathIndex].transform.position - Path[_pathIndex - 1].transform.position).magnitude;
        float completed = (GameObject.transform.position - Path[_pathIndex - 1].transform.position).magnitude;

        float partialScore = completed / total;
        progressScore = (_pathIndex - 1 + partialScore) / Path.Count;
        
        float fitness = progressScore;
        
        Debug.Log("Fitness: " + fitness);
        
        return fitness;
    }

    public GameObject GetTargetNode()
    {
        return Path[_pathIndex];
    }

    public Quaternion GetRotationToTargetNode()
    {
        var rotation = Quaternion.LookRotation(
            (Path[_pathIndex].transform.position - Path[_pathIndex-1].transform.position).normalized,
            Vector3.up
        );

        return rotation;
    }

    private void UpdateAngleToNextNode()
    {
        // angle between vehicle's facing and the direction of the next node
        var angle = Vector3.SignedAngle(GameObject.transform.forward,
            Path[_pathIndex].transform.position - GameObject.transform.position, Vector3.up);

        Debug.Log(angle);
        Controller.guidingAngle = angle / 180;
    }

    public void Update()
    {
        var delta = GetTargetNode().transform.position - GameObject.transform.position;

        if (delta.sqrMagnitude <= _distanceSqrToAchieve)
        {
            if (_pathIndex == Path.Count)
            {
                // destination achieved
                GameObject.SetActive(false);    
            }
            
            _pathIndex += 1;
            UpdateAngleToNextNode();
        }
    }
}

public class EvolutionManager : MonoBehaviour
{
    public Graph graph;
    
    public int populationSize = 15;
    public GameObject prefab;
    public float crossoverChance = 0.5f;
    public float mutationRate = 0.05f;
    public float mutationAmount = 1f;

    public List<int> layerSizes = new() { 6, 5, 4, 2 };
    // activation functions used at each non-input layer (relu, sigmoid, tanh)
    public List<string> activationFuncList = new() { "tanh", "tanh", "tanh" };
    
    private int _aliveVehicleCount;
    private readonly Dictionary<int, Vehicle> _vehicles = new ();
    private readonly System.Random _rand = new ();

    // remove these later
    public GameObject startNode;
    public GameObject endNode;

    private List<Genotype> Genotypes
    {
        get
        {
            var result = new List<Genotype>();

            foreach (var (_, vehicle) in _vehicles)
                result.Add(vehicle.Genotype);

            return result;
        }

        set
        {
            var index = 0;
            
            foreach (var (_, vehicle) in _vehicles)
                vehicle.Genotype = value[index++];
        }
    }
    
    private void Start()
    {
        // Disable collision between vehicles
        int vehicleLayer = LayerMask.NameToLayer("Vehicle");
        Physics.IgnoreLayerCollision(vehicleLayer, vehicleLayer, true);
        
        var genotypes = GenerateGenotypes();
        InstantiateVehicles(genotypes);
        SpawnVehicles();
    }

    private void Update()
    {
        foreach (var (key, vehicle) in _vehicles)
        {
            if (!vehicle.GameObject.activeSelf)
                continue;

            vehicle.Update();
        }
    }

    private List<Genotype> GenerateGenotypes()
    {
        var genotypes = new List<Genotype>();
        
        for (var i = 0; i < populationSize; i++)
        {
            var weights = new List<double>();
            
            for (var j = 0; j < layerSizes.Count - 1; j++)
            {
                weights.AddRange(XavierInit(layerSizes[j], layerSizes[j + 1]));
                weights.AddRange(GenerateGaussianList(layerSizes[j + 1]));
            }

            genotypes.Add(new Genotype(weights, 0));
        }

        return genotypes;
    }

    private void InstantiateVehicles(List<Genotype> genotypes)
    {
        for (var i = 0; i < populationSize; i++)
        {
            var path = graph.GetShortestPath(GetStartNode(), GetEndNode());
            //Debug.Log("Vector List: " + string.Join(", ", path.Select(v => v.name)));
        
            var instance = Instantiate(prefab);
            var vehicle = new Vehicle(genotypes[i], instance, path);
            
            vehicle.Controller.OnHitWall += () =>
            {
                DisableVehicle(vehicle);
            };
            _vehicles.Add(vehicle.ID, vehicle);
        }
    }

    GameObject GetStartNode()
    {
        return startNode;
    }

    GameObject GetEndNode()
    {
        return endNode;
    }
    
    // ReSharper disable Unity.PerformanceAnalysis
    private void SpawnVehicles()
    {
        // allWeights is the list of weights for all vehicles
        // for each vehicle, the weights list format is as follows
        // [layer1->2 weights. layer2 biases, layer2->3 weights, layer3 biases]
        // the list is flattened to 1D

        _aliveVehicleCount = populationSize;

        foreach ((int _, var vehicle) in _vehicles)
        {
            vehicle.Reset();
            
            var genotype = vehicle.Genotype;
            var instance = vehicle.GameObject;

            instance.transform.position = vehicle.Path[0].transform.position;
            instance.transform.rotation = vehicle.GetRotationToTargetNode();

            instance.SetActive(true);

            var weights = new List<List<double>>();
            var biasList = new List<List<double>>(); // biases that will be added to each non-input layer

            var lastIndex = 0;

            for (var j = 0 ; j < layerSizes.Count - 1; j++)
            {
                int weightsLen = layerSizes[j] * layerSizes[j + 1];
                weights.Add(genotype.Weights.GetRange(lastIndex, weightsLen));

                lastIndex += weightsLen;

                biasList.Add(genotype.Weights.GetRange(lastIndex, layerSizes[j + 1]));

                lastIndex += layerSizes[j + 1];
            }

            vehicle.Controller.Fnn = new NN(layerSizes, weights, biasList, activationFuncList);
        }
    }

    private void DisableVehicle(Vehicle vehicle)
    {
        vehicle.Genotype.Fitness = vehicle.GetFitness();
        vehicle.GameObject.SetActive(false);

        _aliveVehicleCount -= 1;

        if (_aliveVehicleCount == 0)
        {
            RespawnVehicles();
        }
    }

    private void RespawnVehicles()
    {
        // reproduction
        var intermediateGeneration = RemainderStochasticSampling(Genotypes);
        var newGeneration = Recombination(intermediateGeneration);
        
        Genotypes = Mutate(newGeneration);

        SpawnVehicles();
    }

    private List<Genotype> RemainderStochasticSampling(List<Genotype> genotypes)
    {
        var generation = new List<Genotype>();

        float fitnessSum = genotypes.Sum(g => g.Fitness);

        foreach (var g in genotypes)
        {
            float expectedCount = g.Fitness / fitnessSum * populationSize;

            var copies = (int)expectedCount;
            float extraCopyChance = expectedCount - copies;

            for (var i = 0; i < copies; i++)
                generation.Add(g);

            if (_rand.NextDouble() < extraCopyChance)
                generation.Add(g);
        }

        return generation;
    }

    private List<Genotype> Recombination(List<Genotype> intermediateGeneration)
    {
        var newGeneration = new List<Genotype>();
        intermediateGeneration.Sort((a, b) => b.Fitness.CompareTo(a.Fitness));

        Debug.Log("Best Fitness in population: " + intermediateGeneration[0].Fitness);
        // keep best 2
        newGeneration.Add(intermediateGeneration[0]);
        newGeneration.Add(intermediateGeneration[1]);

        while (newGeneration.Count < populationSize)
        {
            int ind1 = _rand.Next(0, intermediateGeneration.Count), ind2 = _rand.Next(0, intermediateGeneration.Count);
                
            while (ind1 == ind2)
                ind2 = _rand.Next(0, intermediateGeneration.Count);


            Genotype offspring1, offspring2;
            (offspring1, offspring2) = CompleteCrossover(intermediateGeneration[ind1], intermediateGeneration[ind2]);

            newGeneration.Add(offspring1);

            if (newGeneration.Count < populationSize)
                newGeneration.Add(offspring2);
        }

        return newGeneration;
    }

    private (Genotype, Genotype) CompleteCrossover(Genotype parent1, Genotype parent2)
    {
        List<double> weights1 = new List<double>(),
                     weights2 = new List<double>();

        int weightsCount = parent1.Weights.Count;

        if (parent2.Weights.Count != weightsCount)
            Debug.LogError("Crossover parents have different sizes");

        for (var i = 0; i < weightsCount; i++)
        {
            if (_rand.NextDouble() < crossoverChance)
            {
                weights1.Add(parent2.Weights[i]);
                weights2.Add(parent1.Weights[i]);
            }
            else
            {
                weights1.Add(parent1.Weights[i]);
                weights2.Add(parent2.Weights[i]);
            }
        }
        
        return (new Genotype(weights1, 0), new Genotype(weights2, 0));
    }

    private List<Genotype> Mutate(List<Genotype> generation)
    {
        for (var i = 0; i < generation.Count; i++)
        {
            int weightsCount = generation[i].Weights.Count;
            var weights = generation[i].Weights;
            
            for (var j = 0; j < weightsCount; j++)
            {
                if (_rand.NextDouble() < mutationRate)
                {
                    weights[j] += _rand.NextDouble() * mutationAmount * 2 - mutationAmount;
                }
            }

            generation[i] = new Genotype(weights, 0);
        }

        return generation;
    }

    private List<double> XavierInit(int NIn, int NOut)
    {
        var weights = new List<double>();
        int size = NIn * NOut;

        for (var i = 0; i < size; i++)
        {
            // Uniform random in range [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
            weights.Add(Math.Sqrt(6.0 / (NIn + NOut)) * (_rand.NextDouble() * 2.0 - 1.0));
        }

        return weights;
    }

    private List<double> GenerateGaussianList(int size, double mean = 0.0, double stddev = 1.0)
    {
        var gaussianNumbers = new List<double>();

        for (var i = 0; i < size; i++)
        {
            gaussianNumbers.Add(RandomGaussian(mean, stddev));
        }

        return gaussianNumbers;
    }

    private double RandomGaussian(double mean = 0.0, double stddev = 1.0)
    {
        // Use the Box-Muller transform
        double u1 = 1.0 - _rand.NextDouble(); // Uniform(0,1] random number
        double u2 = 1.0 - _rand.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return z * stddev + mean;
    }
}
