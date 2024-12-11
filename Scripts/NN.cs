using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System.Dynamic;
using System;
using static UnityEngine.Rendering.DebugUI;
using Unity.VisualScripting.FullSerializer;
using System.Collections.Generic;
public class NN
{
    private Vector<double> input = Vector<double>.Build.Dense(6);
    private List<Matrix<double>> Layers = new List<Matrix<double>>();
    private List<Vector<double>> LayerBiases = new List<Vector<double>>();
    private List<int> LayerSizes;
    private List<string> ActivationFuncs;

    public NN(List<int> _LayerSizes, List<List<double>> WeightsList, List<List<double>> BiasesList, List<string> _ActivationFuncs)
    {
        LayerSizes = _LayerSizes;
        ActivationFuncs = _ActivationFuncs;
        for(int i = 0; i < LayerSizes.Count - 1; i++) 
        {
            Matrix<double> M = ArrayToMatrix(LayerSizes[i], LayerSizes[i + 1], WeightsList[i]);
            Layers.Add(M);
            
            Vector<double> BiasesVector = Vector<double>.Build.DenseOfEnumerable(BiasesList[i]);
            LayerBiases.Add(BiasesVector);
        }

        if (Layers.Count != LayerBiases.Count)
            throw new ArgumentException("Inconsistent layer Biases");

        if (Layers.Count + 1 != LayerSizes.Count)
            throw new ArgumentException("Inconsistent layer construction");

        if (Layers.Count != ActivationFuncs.Count)
            throw new ArgumentException("Inconsistent layer activation num");
    }

    public double[] ForwardPass()
    {
        Vector<double> tmp_in = input;

        for (int i = 0; i < Layers.Count; i++) 
        {
            Func<double, double> activationFunc = ActivationFuncs[i] switch
            {
                "sigmoid" => Sigmoid,
                "relu" => ReLU,
                "tanh" => Tanh,
                _ => throw new ArgumentException($"Unsupported Activation Function: {ActivationFuncs[i]}")
            };

            tmp_in = HiddenLayerPass(tmp_in, LayerSizes[i + 1], Layers[i], LayerBiases[i], activationFunc);
        }

        return tmp_in.ToArray();
    }

    public void SetInput(double[] values)
    {
        if (values.Length != input.Count)
            throw new ArgumentException($"Input array must have {input.Count} elements.");

        for (int j = 0; j < input.Count; j++)
        {
            input[j] = values[j];
        }
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private static double ReLU(double x) => Math.Max(0, x);
    private static double Tanh(double x) => Math.Tanh(x);

    private Matrix<double> ArrayToMatrix(int rows, int cols, List<double> Weights)
    {
        if (Weights.Count != rows * cols)
            throw new ArgumentException($"Weights must have {rows * cols} elements.");

        Matrix<double> WeightMatrix = Matrix<double>.Build.DenseOfRowMajor(rows, cols, Weights);
        return WeightMatrix;
    }
    private Vector<double> HiddenLayerPass(Vector<double> Input, int OutputSize, Matrix<double>WeightMatrix, Vector<double> Biases,
                                       Func<double, double> ActivationFunction)
    {
        int InputSize = Input.Count;
        if (Biases.Count != OutputSize)
            throw new ArgumentException($"Biases must have {OutputSize} elements.");

        Vector<double> PreActivation = Input * WeightMatrix + Biases;

        Vector<double> ActivatedOutput = PreActivation.Map(ActivationFunction);

        return ActivatedOutput;
    }
}
