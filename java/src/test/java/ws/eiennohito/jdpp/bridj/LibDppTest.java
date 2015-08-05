package ws.eiennohito.jdpp.bridj;

import org.bridj.Pointer;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author eiennohito
 * @since 15/08/05
 */
public class LibDppTest {

  @Test
  public void somethingIsCreated() {
    Pointer<Double> dbls = Pointer.allocateDoubles(4);
    dbls.getDoubleBuffer().put(new double[]{1, 2, 5, 9});
    Pointer<?> handle = LibDpp.dpp_create_c_kernel(dbls, 2, 2);
    assertNotNull(handle);
    assertNotEquals(0L, handle.getPeer());
    LibDpp.dpp_delete_c_kernel(handle);
    dbls.release();
  }

  @Test
  public void smallSelection() {
    Pointer<Double> doubles = Pointer.allocateDoubles(12);
    doubles.getDoubleBuffer().put(new double[]{
        0, 1, 0,
        1, 1, 1,
        1, 0, 0,
        0, 0, 1
    });

    Pointer<?> handle = LibDpp.dpp_create_c_kernel(doubles, 3, 4);
    Pointer<Long> res = Pointer.allocateLongs(4);
    LibDpp.dpp_select_from_c(handle, 1, res);

    assertEquals(1, res.getLongAtIndex(0));

    doubles.release();
    res.release();
    LibDpp.dpp_delete_c_kernel(handle);
  }

}
