using UnityEngine;
using System.Collections;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using MathNet.Numerics.Distributions;
using System;

public class VehicleController : MonoBehaviour
{
    public event System.Action HitWall;

    private const float MAX_DISTANCE = 10000;
    private const float MAX_VEL = 20f;
    private const float ACCELERATION = 8f;
    private const float TURN_SPEED = 100;
    
    bool UseUserInput = false;

    private float VEL_FRICT = 1f;

    public float Velocity
    {
        get;
        private set;
    }

    public Quaternion Rotation
    {
        get;
        private set;
    }

    private double gasInput, turningInput;
    private NN fnn;

    void Start()
    {
        List<double> weight1 = XavierInit(5, 4);
        List<double> weight2 = XavierInit(4, 3);
        //weight1 = new List<double>(new double[] { 0, 0, 0, 0,
        //                                          0, 0, 0, 0,
        //                                          1, 0, 0, 0,
        //                                          0, 0, 0, 0,
        //                                          0, 0, 0, 0,});
        //weight2 = new List<double>(new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } );   
        List<List<double>> weightslist = new List<List<double>> { weight1, weight2 };
        List<List<double>> biaslist = new List<List<double>> { new() { 0, 0, 0, 0 }, new() { 0, 0, 0 } };
        fnn = new NN(new List<int> { 5, 4, 3 }, 
                     weightslist, 
                     biaslist, 
                     new List<string> { "tanh", "tanh" });
    }
    
    List<double> XavierInit(int NIn, int NOut)
    {
        System.Random rand = new System.Random();
        List<double> weights = new List<double>();
        int size = NIn * NOut;
        for (int i = 0; i < size ; i++)
        {
            weights.Add(System.Math.Sqrt(6.0 / (NIn + NOut)) * (rand.NextDouble() * 2.0 - 1.0)); // Uniform random in range [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
        }

        return weights;
    }
    void Update()
    {
        if (UseUserInput)
            GetInput();
    }

    void FixedUpdate ()
    {
        if (!UseUserInput)
            GetInput();

        ApplyInput();

        ApplyVelocity();

        ApplyFriction();
    }

    private void GetInput()
    {
        if (UseUserInput)
        {
            if (Input.GetKeyDown("w"))
                gasInput = 1.0;
            else if (Input.GetKeyUp("w"))
                gasInput = 0.0;
            else if (Input.GetKeyDown("a"))
                turningInput = 1.0;
            else if (Input.GetKeyUp("a"))
                turningInput = 0.0;
            else if (Input.GetKeyDown("d"))
                turningInput = -1.0;
            else if (Input.GetKeyUp("d"))
                turningInput = 0.0;
            else if (Input.GetKeyDown("s"))
                VEL_FRICT = 20f;
            else if (Input.GetKeyUp("s"))
                VEL_FRICT = 1f;
        }
        else
        {
            double[] distances = new double[] { DistInDir(new Vector3(-1, 0, 0)),
                                                DistInDir(new Vector3(-1, 0, 1)),
                                                DistInDir(new Vector3(0, 0, 1)),
                                                DistInDir(new Vector3(1, 0, 1)),
                                                DistInDir(new Vector3(1, 0, 0))};
            Debug.Log("ditances: " + string.Join(", ", distances));
            fnn.SetInput(distances);

            double[] outputs = fnn.ForwardPass();
            gasInput = outputs[0];
            turningInput = outputs[1];
            VEL_FRICT = Mathf.Max(1f, 40 * (float)outputs[2]);
        }
    }

    private double DistInDir(Vector3 dir)
    {
        Vector3 worldDirection = transform.TransformDirection(dir.normalized);

        Debug.DrawRay(transform.position, worldDirection * 5, UnityEngine.Color.red);
        Ray ray = new Ray(transform.position + new Vector3(0, 0.1f, 0), worldDirection);

        if (Physics.Raycast(ray, out RaycastHit hit))
            return hit.distance;
        else
            return MAX_DISTANCE; //just some arbitrary number atm, need to change later
        
    }

    private void ApplyInput()
    {
        if (gasInput > 1)
            gasInput = 1;
        else if (gasInput < 0)
            gasInput = 0;
        
        if (turningInput > 1)
            turningInput = 1;
        else if (turningInput < -1)
            turningInput = -1;

        bool canAccelerate = false;
        if (gasInput < 0)
            canAccelerate = Velocity > gasInput * MAX_VEL;
        else if (gasInput > 0)
            canAccelerate = Velocity < gasInput * MAX_VEL;
        
        if (canAccelerate)
        {
            Velocity += (float)gasInput * ACCELERATION * Time.deltaTime;

            if (Velocity > MAX_VEL)
                Velocity = MAX_VEL;
            else if (Velocity < -MAX_VEL)
                Velocity = -MAX_VEL;
        }

        Rotation = transform.rotation;
        Rotation *= Quaternion.AngleAxis((float)-turningInput * TURN_SPEED * Time.deltaTime, new Vector3(0, 1, 0));
    }

    private void ApplyVelocity()
    {
        Vector3 direction = new Vector3(0, 0, 1);
        transform.rotation = Rotation;
        direction = Rotation * direction;

        this.transform.position += direction * Velocity * Time.deltaTime;
    }

    private void ApplyFriction()
    {
        if (gasInput == 0)
        {
            if (Velocity > 0)
            {
                Velocity -= VEL_FRICT * Time.deltaTime;
                if (Velocity < 0)
                    Velocity = 0;
            }
            else if (Velocity < 0)
            {
                Velocity += VEL_FRICT * Time.deltaTime;
                if (Velocity > 0)
                    Velocity = 0;            
            }
        }
    }

}
