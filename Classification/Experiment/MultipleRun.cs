using Classification.Performance;

namespace Classification.Experiment
{
    public interface MultipleRun
    {
        ExperimentPerformance Execute(Experiment experiment);
    }
}