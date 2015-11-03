package ws.eiennohito.jdpp;

import org.bridj.Pointer;
import ws.eiennohito.jdpp.bridj.LibDpp;

import java.io.Closeable;
import java.io.IOException;
import java.nio.DoubleBuffer;
import java.nio.LongBuffer;

/**
 * @author eiennohito
 * @since 15/08/05
 */
public class CSelector implements Closeable {

  private Pointer<Double> matrix;
  private Pointer<Long> sampleResult;
  private long ndim;
  private long size;


  CSelector(Pointer<Double> matrix, Pointer<Long> sampleResult, long ndim, long size) {
    this.matrix = matrix;
    this.sampleResult = sampleResult;
    this.ndim = ndim;
    this.size = size;
  }

  /*package*/ DoubleBuffer buffer() {
    return matrix.getDoubleBuffer();
  }

  /*package*/ LongBuffer sample(long max) {
    return sample(max, size);
  }

  /*package*/ LongBuffer sample(long max, long curSize) {
    Pointer<?> handle = LibDpp.dpp_create_c_kernel(matrix, ndim, curSize);
    LibDpp.dpp_select_from_c(handle, max, sampleResult);
    LibDpp.dpp_delete_c_kernel(handle);
    LongBuffer longBuffer = sampleResult.getLongBuffer();
    longBuffer.limit((int) max);
    return longBuffer.slice();
  }

  @Override
  public void close() throws IOException {
    tryFree(matrix);
    tryFree(sampleResult);
  }

  private void tryFree(Pointer<?> ptr) {
    ptr.release();
  }

  public static CSelector forSize(long dim, long items) {
    Pointer<Double> matrix = Pointer.allocateDoubles(dim * items);
    Pointer<Long> retvals = Pointer.allocateLongs(items);
    return new CSelector(matrix, retvals, dim, items);
  }

  public VectorFiller filler() { return new FillerImpl(); }

  class FillerImpl implements VectorFiller {
    int position = 0;

    private void checkCanInsert() {
      if (position >= size)
        throw new DppException("can not insert " + position + " element to buffer, have only " + size + " entries");
    }

    @Override
    public int fillDouble(double[] data, int start, int length) {
      checkCanInsert();
      checkLength(length);
      long offset = ndim * position * 8;
      matrix.setDoublesAtOffset(offset, data, start, length);
      return position++;
    }

    private void checkLength(int length) {
      if (length != ndim) {
        throw new DppException("data size is of invalid size: " + length + ", but was waiting " + ndim);
      }
    }

    private double[] buffer = new double[(int)ndim];

    @Override
    public int fillFloat(float[] data, int start, int length) {
      checkCanInsert();
      checkLength(length);
      return convertAndFill(buffer, data, start, length);
    }

    private int convertAndFill(double[] buffer, float[] data, int start, int length) {
      int totLen = length - start;
      for (int i = 0; i < totLen; ++i) {
        buffer[i] = data[i + start];
      }
      return fillDouble(buffer, 0, totLen);
    }

    @Override
    public int sample(long[] result) {
      LongBuffer lbuf = CSelector.this.sample(result.length, position);
      lbuf.get(result);
      return lbuf.position();
    }

    @Override
    public void reset() {
      position = 0;
    }
  }

}
