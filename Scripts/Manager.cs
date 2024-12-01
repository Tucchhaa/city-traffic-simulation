using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class Manager : MonoBehaviour
{
    public int population = 15;
    public GameObject prefab;
    public float CrossoverChance = 0.5f;
    public float MutationRate = 0.05f;
    public float MutationAmount = 1f;

    private Transform spawnPoint;

    private List<GameObject> vehicles = new List<GameObject>();
    public List<int> layerSizes = new List<int> { 5, 4, 3 };

    // activation functions used at each non-input layer (relu, sigmoid, tanh)
    public List<string> activationFuncList = new List<string> { "tanh", "tanh" };

    private int currentVehicleNum;

    public struct Genotype
    {
        public List<double> weights;
        public float fitness;

        public Genotype(List<double> _weights, float _fitness)
        {
            weights = _weights;
            fitness = _fitness;
        }
    }

    private List<Genotype> genotypes = new List<Genotype>();

    private System.Random rand = new System.Random();

    private void Start()
    {
        currentVehicleNum = population;
        int vehicleLayer = LayerMask.NameToLayer("VehicleSelf"); // Get the layer by name
        Physics.IgnoreLayerCollision(vehicleLayer, vehicleLayer, true);

        spawnPoint = transform;

        for (int i = 0; i < population; i++)
        {
            List<double> w = new List<double>();
            for (int j = 0; j < layerSizes.Count - 1; j++)
            {
                w.AddRange(XavierInit(layerSizes[j], layerSizes[j + 1]));
                w.AddRange(GenerateGaussianList(layerSizes[j + 1]));
            }

            genotypes.Add(new Genotype(w, 0));
        }

        SpawnVehicles();
    }

    private void SpawnVehicles()
    {
        // allWeights is the list of weights for all vehicles
        // for each vehicle, the weights list format is as follows
        // [layer1->2 weights. layer2 biases, layer2->3 weights, layer3 biases]
        // the list is flattened to 1D

        for (int i = 0; i < population; i++)
        {
            GameObject vehicle = Instantiate(prefab, spawnPoint.position, spawnPoint.rotation);
            vehicles.Add(vehicle);

            vehicle.GetComponent<VehicleController>().manager = this; // Assign manager to vehicle
            vehicle.GetComponent<VehicleController>().index_in_list = i;

            var weightsList = new List<List<double>>();
            var biasList = new List<List<double>>(); // biases that will be added to each non-input layer

            int lastInd = 0;

            for (int j = 0 ; j < layerSizes.Count - 1; j++)
            {
                int weightsLen = layerSizes[j] * layerSizes[j + 1];
                weightsList.Add(genotypes[i].weights.GetRange(lastInd, weightsLen));

                lastInd += weightsLen;

                biasList.Add(genotypes[i].weights.GetRange(lastInd, layerSizes[j + 1]));

                lastInd += layerSizes[j + 1];
            }

            vehicle.GetComponent<VehicleController>()._fnn = new NN(layerSizes, weightsList, biasList, activationFuncList);
        }
    }

    public void VehicleDied(GameObject vehicle, float fitness)
    {
        int index = vehicle.GetComponent<VehicleController>().index_in_list;
        genotypes[index] = new Genotype(genotypes[index].weights, fitness);

        vehicles.Remove(vehicle);
        Destroy(vehicle);

        if (vehicles.Count == 0)
        {
            StartCoroutine(RespawnVehicles());
        }
    }

    IEnumerator RespawnVehicles()
    {
        yield return new WaitForSeconds(1f);

        // reproduction
        List<Genotype> intermediateGeneration = RemainderStochasticSampling();
        List<Genotype> newGeneration = Recombination(intermediateGeneration);
        genotypes = Mutate(newGeneration);

        // respawn with new weights
        SpawnVehicles();
    }

    List<Genotype> RemainderStochasticSampling()
    {
        List<Genotype> intermediateGeneration = new List<Genotype>();

        float fitnessSum = 0;
        foreach (Genotype g in genotypes)
            fitnessSum += g.fitness;

        foreach (Genotype g in genotypes)
        {
            float expected_count = g.fitness / fitnessSum * population;

            int copies = (int)expected_count;
            float chance_for_extra_copies = expected_count - copies;

            for (int i = 0; i < copies; i++)
                intermediateGeneration.Add(g);

            if (rand.NextDouble() < chance_for_extra_copies)
                intermediateGeneration.Add(g);
        }

        return intermediateGeneration;
    }

    List<Genotype> Recombination(List<Genotype> intermediateGeneration)
    {
        List<Genotype> newGeneration = new List<Genotype>();
        intermediateGeneration.Sort((a, b) => b.fitness.CompareTo(a.fitness));

        Debug.Log("Best Fitness in population: " + intermediateGeneration[0].fitness);
        // keep best 2
        newGeneration.Add(intermediateGeneration[0]);
        newGeneration.Add(intermediateGeneration[1]);

        while (newGeneration.Count < population)
        {
            int ind1 = rand.Next(0, intermediateGeneration.Count), ind2 = rand.Next(0, intermediateGeneration.Count);
                
            while (ind1 == ind2)
                ind2 = rand.Next(0, intermediateGeneration.Count);


            Genotype offspring1, offspring2;
            (offspring1, offspring2) = CompleteCrossover(intermediateGeneration[ind1], intermediateGeneration[ind2]);

            newGeneration.Add(offspring1);

            if (newGeneration.Count < population)
                newGeneration.Add(offspring2);
        }

        return newGeneration;
    }

    private (Genotype, Genotype) CompleteCrossover(Genotype parent1, Genotype parent2)
    {
        List<double> weights1 = new List<double>(),
                     weights2 = new List<double>();

        int genotype_len = parent1.weights.Count;

        if (parent2.weights.Count != genotype_len)
            Debug.LogError("Crossover parents have different sizes");

        for (int i = 0; i < genotype_len; i++)
        {
            if (rand.NextDouble() < CrossoverChance)
            {
                weights1.Add(parent2.weights[i]);
                weights2.Add(parent1.weights[i]);
            }
            else
            {
                weights1.Add(parent1.weights[i]);
                weights2.Add(parent2.weights[i]);
            }
        }
        return (new Genotype(weights1, 0), new Genotype(weights2, 0));
    }

    private List<Genotype> Mutate(List<Genotype> generation)
    {
        for (int i = 0; i < generation.Count; i++)
        {
            int w_size = generation[i].weights.Count;
            List<double> weights = generation[i].weights;
            for (int j = 0; j < w_size; j++)
            {
                if (rand.NextDouble() < MutationRate)
                {
                    weights[j] += rand.NextDouble() * MutationAmount * 2 - MutationAmount;
                }
            }

            generation[i] = new Genotype(weights, 0);
        }

        return generation;
    }

    private List<double> XavierInit(int NIn, int NOut)
    {
        List<double> weights = new List<double>();
        int size = NIn * NOut;

        for (int i = 0; i < size; i++)
        {
            // Uniform random in range [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
            weights.Add(Math.Sqrt(6.0 / (NIn + NOut)) * (rand.NextDouble() * 2.0 - 1.0));
        }

        return weights;
    }

    private List<double> GenerateGaussianList(int size, double mean = 0.0, double stddev = 1.0)
    {
        List<double> gaussianNumbers = new List<double>();

        for (int i = 0; i < size; i++)
        {
            gaussianNumbers.Add(RandomGaussian(mean, stddev));
        }

        return gaussianNumbers;
    }

    private double RandomGaussian(double mean = 0.0, double stddev = 1.0)
    {
        // Use the Box-Muller transform
        double u1 = 1.0 - rand.NextDouble(); // Uniform(0,1] random number
        double u2 = 1.0 - rand.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return z * stddev + mean;
    }
}
