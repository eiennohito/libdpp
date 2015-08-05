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

  public DoubleBuffer buffer() {
    return matrix.getDoubleBuffer();
  }

  public LongBuffer sample(long max) {
    Pointer<?> handle = LibDpp.dpp_create_c_kernel(matrix, ndim, size);
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
}
