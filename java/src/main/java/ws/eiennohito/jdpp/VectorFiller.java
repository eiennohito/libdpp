package ws.eiennohito.jdpp;

/**
 * @author eiennohito
 * @since 2015/11/03
 */
public interface VectorFiller {
  int fillDouble(double[] data, int start, int length);
  int fillFloat(float[] data, int start, int length);
  int sample(long[] result);
  void reset();
}
