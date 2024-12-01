using UnityEngine;
using System.Collections.Generic;
using System;

public class VehicleController : MonoBehaviour
{
    private const float MaxDistance = 10000;
    private const float DebugRayLen = 5f;

    public float maxVelocity = 20f;
    public float accelerationFactor = 8f;
    public float turnFactor = 100;
    public float frictionFactor = 40f;
    public Vector3 rayBias = new (0, 0.1f, 0);
    public int index_in_list;

    public NN _fnn;
    private float _velocity;
    private Quaternion _rotation;
    public LayerMask rayIgnore;

    public Manager manager = null;

    public float fitness = 0;

    private DateTime startTime;

    private void Start()
    {
        rayIgnore = LayerMask.GetMask("VehicleSelf");
        startTime = DateTime.Now;
    }
    
    private void FixedUpdate ()
    {
        var (gas, turning, friction) = GetNnOutput();

        MoveVehicle(gas, turning, friction);
    }

    private void OnCollisionEnter(Collision collision)
    {
        // If collider is not the ground (Quad)
        if (collision.collider.name != "Quad")
        {
            DateTime endTime = DateTime.Now;
            TimeSpan interval = endTime - startTime;

            double seconds = interval.TotalSeconds;
            fitness = (float)seconds;
            manager.VehicleDied(gameObject, fitness);
            this.enabled = false;
        }
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
        
        //Debug.Log("ditances: " + string.Join(", ", distances));
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

        if (Physics.Raycast(ray, out RaycastHit hit, MaxDistance, ~rayIgnore))
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
            // _velocity = Mathf.Clamp(_velocity, -maxVelocity, maxVelocity);

            // needs further adjustments
            _velocity = Mathf.Clamp(_velocity, 0.5f * maxVelocity, maxVelocity);
        }

        _rotation = transform.rotation;
        _rotation *= Quaternion.AngleAxis(-turn * turnFactor * Time.deltaTime, Vector3.up);
        
        var direction = _rotation * Vector3.forward;

        transform.position += direction * (_velocity * Time.deltaTime);
        transform.rotation = _rotation;
    }
}
