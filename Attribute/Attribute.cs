using System.Collections.Generic;

namespace Classification.Attribute
{
    public abstract class Attribute
    {
        public abstract int ContinuousAttributeSize();
        public abstract List<double> ContinuousAttributes();
    }
}