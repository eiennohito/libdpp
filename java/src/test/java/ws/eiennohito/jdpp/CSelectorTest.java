package ws.eiennohito.jdpp;

import org.junit.Test;

import java.io.IOException;
import java.nio.LongBuffer;

import static org.junit.Assert.*;

/**
 * @author eiennohito
 * @since 15/08/05
 */
public class CSelectorTest {
  @Test
  public void testSelects() throws IOException {
    CSelector selector = CSelector.forSize(3, 4);
    try {
      selector.buffer().put(new double[]{
          0, 0, 1,
          0, 1, 0,
          1, 0, 0,
          1, 1, 1
      });
      LongBuffer buf = selector.sample(1);
      assertEquals(1, buf.remaining());
      assertEquals(3, buf.get(0));
    } finally {
      selector.close();
    }
  }

  @Test
  public void fillerWorks() throws IOException {
    CSelector selector = CSelector.forSize(3, 4);
    try {
      VectorFiller filler = selector.filler();
      filler.fillFloat(new float[]{0, 0, 1}, 0, 3);
      filler.fillFloat(new float[]{0, 1, 0}, 0, 3);
      filler.fillFloat(new float[]{1, 0, 0}, 0, 3);
      filler.fillFloat(new float[]{1, 1, 1}, 0, 3);
      long[] res = new long[1];
      assertEquals(1, filler.sample(res));
      assertEquals(3, res[0]);
    } finally {
      selector.close();
    }
  }
}
