namespace Classification.Experiment
{
    public interface SingleRun
    {
        Performance.Performance Execute(Experiment experiment);
    }
}