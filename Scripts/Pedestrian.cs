using UnityEngine;
using System.Collections;
using UnityEngine.Serialization;

public class Pedestrian : MonoBehaviour
{
    public GameObject startPos;
    public GameObject endPos;
    public GameObject pedestrianObject;
    
    public float speed = 2.0f;
    public float waitTimeFloor = 3f;
    public float waitTimeCeil = 10f;
    
    private Vector3 _from;
    private Vector3 _to;

    private void Start()
    {
        _from = startPos.transform.position;
        _to = endPos.transform.position;
        pedestrianObject.transform.position = _from;
        StartCoroutine(PedestrianMovementRoutine());
    }

    private IEnumerator PedestrianMovementRoutine()
    {
        while (true)
        {
            var waitTime = Random.Range(waitTimeFloor, waitTimeCeil);
            yield return new WaitForSeconds(waitTime);

            yield return StartCoroutine(MovePedestrian());
        }
    }

    private IEnumerator MovePedestrian()
    {
        pedestrianObject.transform.position = _from;

        var distance = Vector3.Distance(_from, _to);
        var duration = distance / speed;
        var elapsed = 0f;

        while (elapsed < duration)
        {
            pedestrianObject.transform.position = Vector3.Lerp(_from, _to, elapsed / duration);

            elapsed += Time.deltaTime;

            yield return null;
        }

        pedestrianObject.transform.position = _to;

        (_to, _from) = (_from, _to);
    }
}
