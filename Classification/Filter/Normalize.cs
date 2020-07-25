using Classification.Attribute;

namespace Classification.Filter
{
    public class Normalize : FeatureFilter
    {
        private readonly Instance.Instance _averageInstance;
        private readonly Instance.Instance _standardDeviationInstance;

        /**
         * <summary> Constructor for normalize feature filter. It calculates and stores the mean (m) and standard deviation (s) of
         * the sample.</summary>
         *
         * <param name="dataSet">Instances whose continuous attribute values will be normalized.</param>
         */
        public Normalize(DataSet.DataSet dataSet) : base(dataSet)
        {
            _averageInstance = dataSet.GetInstanceList().Average();
            _standardDeviationInstance = dataSet.GetInstanceList().StandardDeviation();
        }

        /**
         * <summary> Normalizes the continuous attributes of a single instance. For all i, new x_i = (x_i - m_i) / s_i.</summary>
         *
         * <param name="instance">Instance whose attributes will be normalized.</param>
         */
        protected override void ConvertInstance(Instance.Instance instance)
        {
            for (var i = 0; i < instance.AttributeSize(); i++)
            {
                if (instance.GetAttribute(i) is ContinuousAttribute) {
                    var xi = (ContinuousAttribute) instance.GetAttribute(i);
                    var mi = (ContinuousAttribute) _averageInstance.GetAttribute(i);
                    var si = (ContinuousAttribute) _standardDeviationInstance.GetAttribute(i);
                    xi.SetValue((xi.GetValue() - mi.GetValue()) / si.GetValue());
                }
            }
        }

        protected override void ConvertDataDefinition()
        {
        }
    }
}