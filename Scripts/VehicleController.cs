using UnityEngine;
using System.Collections.Generic;
using System;

public class VehicleController : MonoBehaviour
{
    public event System.Action HitWall;

    private const float MaxDistance = 10000;
    private const float DebugRayLen = 5f;

    public float maxVelocity = 20f;
    public float accelerationFactor = 8f;
    public float turnFactor = 100;
    public float frictionFactor = 40f;
    public Vector3 rayBias = new (0, 0.1f, 0);

    private NN _fnn;
    private float _velocity;
    private Quaternion _rotation;

    private void Start()
    {
        var layerSizes = new List<int> { 5, 4, 3 };
        var weightsList = new List<List<double>>
        {
            XavierInit(5, 4), // weights to map layer 1 to layer 2 (5 -> 4)
            XavierInit(4, 3)  // weights to map layer 2 to layer 3 (4 -> 3)
        };
        
        // biases that will be added to each non-input layer [4, 3]
        var biasList = new List<List<double>>
        {
            new() { 0, 0, 0, 0 }, 
            new() { 0, 0, 0 }
        };
        
        // activation functions used at each non-input layer (relu, sigmoid, tanh)
        var activationFuncList = new List<string> { "tanh", "tanh" };
        
        _fnn = new NN(layerSizes, weightsList, biasList, activationFuncList); 
    }
    
    private List<double> XavierInit(int NIn, int NOut)
    {
        System.Random rand = new System.Random();
        List<double> weights = new List<double>();
        int size = NIn * NOut;
        
        for (int i = 0; i < size ; i++)
        {
            // Uniform random in range [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
            weights.Add(Math.Sqrt(6.0 / (NIn + NOut)) * (rand.NextDouble() * 2.0 - 1.0));
        }

        return weights;
    }
    
    private void FixedUpdate ()
    {
        var (gas, turning, friction) = GetNnOutput();

        MoveVehicle(gas, turning, friction);
    }

    private (float, float, float) GetNnOutput()
    {
        var distances = new double[] { 
            GetDirectionDistance(new Vector3(-1, 0, 0)),
            GetDirectionDistance(new Vector3(-1, 0, 1)),
            GetDirectionDistance(new Vector3(0, 0, 1)),
            GetDirectionDistance(new Vector3(1, 0, 1)),
            GetDirectionDistance(new Vector3(1, 0, 0))
        };
        
        // Debug.Log("ditances: " + string.Join(", ", distances));
        _fnn.SetInput(distances);

        var outputs = _fnn.ForwardPass();
        var gas  = (float)outputs[0];
        var turn = (float)outputs[1];
        var friction = Mathf.Max(1f, frictionFactor * (float)outputs[2]);

        return (gas, turn, friction);
    }

    private double GetDirectionDistance(Vector3 dir)
    {
        Vector3 worldDirection = transform.TransformDirection(dir.normalized);

        Debug.DrawRay(transform.position, worldDirection * DebugRayLen, Color.red);
        
        // Add ray bias to make sure ray doesn't hit the ground
        Ray ray = new Ray(transform.position + rayBias, worldDirection);

        if (Physics.Raycast(ray, out RaycastHit hit))
            return hit.distance;
        
        return MaxDistance;
    }

    private void MoveVehicle(float gas, float turn, float friction)
    {
        Math.Clamp(gas, 0, 1);
        Math.Clamp(turn, -1, 1);
        
        if (gas == 0) {
            _velocity = Mathf.MoveTowards(_velocity, 0, friction * Time.deltaTime);
        }
        else
        {
            _velocity += gas * accelerationFactor * Time.deltaTime;
            _velocity = Mathf.Clamp(_velocity, -maxVelocity, maxVelocity);
        }

        _rotation = transform.rotation;
        _rotation *= Quaternion.AngleAxis(-turn * turnFactor * Time.deltaTime, Vector3.up);
        
        var direction = _rotation * Vector3.forward;

        transform.position += direction * (_velocity * Time.deltaTime);
        transform.rotation = _rotation;
    }
}
