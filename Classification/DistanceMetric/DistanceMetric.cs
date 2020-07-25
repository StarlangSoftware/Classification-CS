namespace Classification.DistanceMetric
{
    public interface DistanceMetric
    {
        double Distance(Instance.Instance instance1, Instance.Instance instance2);
    }
}