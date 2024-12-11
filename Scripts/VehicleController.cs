using UnityEngine;
using System.Collections.Generic;
using System;

public class VehicleController : MonoBehaviour
{
    public event Action OnHitWall;
    
    private const float MaxRaycastDistance = 10000;

    public float maxVelocity = 20f;
    public float accelerationFactor = 8f;
    public float turnFactor = 100;
    public float frictionFactor = 40f;
    public Vector3 rayBias = new (0, 0.5f, 0);
    public LayerMask rayMask = ~0;
    public float guidingAngle = 0;
    public NN Fnn;
    private float _velocity;
    private Quaternion _rotation;
    private Collider _collider;

    private void Start()
    {
        _collider = GetComponent<Collider>();
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
        
        OnHitWall?.Invoke();
    }

    private (float, float, float) GetNnOutput()
    {
        var distances = new double[] { 
            GetDirectionDistance(new Vector3(-1, 0, 0)),
            GetDirectionDistance(new Vector3(-1, 0, 1)),
            GetDirectionDistance(new Vector3(0, 0, 1)),
            GetDirectionDistance(new Vector3(1, 0, 1)),
            GetDirectionDistance(new Vector3(1, 0, 0)),
            guidingAngle
        };
        
        //Debug.Log("ditances: " + string.Join(", ", distances));
        Fnn.SetInput(distances);

        var outputs = Fnn.ForwardPass();
        var gas_and_friction  = (float)outputs[0];
        var turn = (float)outputs[1];
        //var friction = Mathf.Max(1f, frictionFactor * (float)outputs[2]);

        var gas = 0.0f;
        var friction = 0.0f;

        if (gas_and_friction < 0)
            friction = Mathf.Max(1f, frictionFactor * gas_and_friction);
        else
            gas = gas_and_friction;

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

        float distance = Physics.Raycast(ray, out var hit, MaxRaycastDistance, rayMask)
            ? hit.distance
            : MaxRaycastDistance;
        
        Debug.DrawRay(rayPoint, worldDirection * distance, Color.green);

        _collider.enabled = true;

        return distance;
    }

    private void MoveVehicle(float gas, float turn, float friction)
    {
        //Math.Clamp(gas, 0, 1);
        //Math.Clamp(turn, -1, 1);

        if (gas == 0) {
            _velocity = Mathf.MoveTowards(_velocity, 0, friction * Time.deltaTime);
        }
        else
        {
            _velocity += gas * accelerationFactor * Time.deltaTime;
            // _velocity = Mathf.Clamp(_velocity, -maxVelocity, maxVelocity);

            // needs further adjustments
        }
        _velocity = Mathf.Clamp(_velocity, 0.5f * maxVelocity, maxVelocity);

        _rotation = transform.rotation;
        _rotation *= Quaternion.AngleAxis(-turn * turnFactor * Time.deltaTime, Vector3.up);
        
        var direction = _rotation * Vector3.forward;

        transform.position += direction * (_velocity * Time.deltaTime);
        transform.rotation = _rotation;
    }
}
