package ws.eiennohito.jdpp;

import org.junit.Test;

import java.nio.LongBuffer;

import static org.junit.Assert.*;

/**
 * @author eiennohito
 * @since 15/08/05
 */
public class CSelectorTest {
  @Test
  public void testSelects() {
    CSelector selector = CSelector.forSize(3, 4);
    selector.buffer().put(new double[]{
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1
    });
    LongBuffer buf = selector.sample(1);
    assertEquals(1, buf.remaining());
    assertEquals(3, buf.get(0));
  }
}
