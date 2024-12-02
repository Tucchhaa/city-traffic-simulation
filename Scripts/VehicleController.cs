using UnityEngine;
using System.Collections.Generic;
using System;

public class VehicleController : MonoBehaviour
{
    private const float MaxRaycastDistance = 10000;
    private const float DebugRayLen = 50f;

    public float maxVelocity = 20f;
    public float accelerationFactor = 8f;
    public float turnFactor = 100;
    public float frictionFactor = 40f;
    public Vector3 rayBias = new (0, 0.5f, 0);
    public int index_in_list;

    public NN _fnn;
    private float _velocity;
    private Quaternion _rotation;
    private Collider _collider;

    public Manager manager = null;

    public float fitness = 0;

    private DateTime startTime;

    private void Start()
    {
        _collider = GetComponent<Collider>();
        startTime = DateTime.Now;
    }
    
    private void FixedUpdate()
    {
        var (gas, turning, friction) = GetNnOutput();
        
        MoveVehicle(gas, turning, friction);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name == "Ground")
            return;
        
        fitness = (float)(DateTime.Now - startTime).TotalSeconds;
        enabled = false;
        
        manager.VehicleDied(gameObject, fitness);
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

    private double GetDirectionDistance(Vector3 direction)
    {
        var worldDirection = transform.TransformDirection(direction.normalized);

        // Disable own collider, so raycast wouldn't hit it
        _collider.enabled = false;
        
        // Add ray bias to make sure ray doesn't hit the ground
        var rayPoint = transform.position + rayBias;
        var ray = new Ray(rayPoint, worldDirection);

        var distance = Physics.Raycast(ray, out var hit, MaxRaycastDistance)
            ? hit.distance
            : MaxRaycastDistance;
        
        Debug.DrawRay(rayPoint, worldDirection * distance, Color.green);

        _collider.enabled = true;

        return distance;
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
