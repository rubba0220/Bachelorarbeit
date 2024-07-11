
import jax
import jax.numpy as jnp
from jax import lax

# Definiere die Funktion, die auf jedem Slice ausgeführt werden soll
def operation_on_slice(carry, slice_array):
    slice_sum = jnp.sum(slice_array)
    return carry, slice_sum

# Hauptfunktion, die die Schleife mit lax.scan implementiert
def process_slices_scan(array, slice_size):
    num_slices = array.shape[0] // slice_size
    slices = array.reshape((num_slices, slice_size))
    
    carry = None
    _, results = lax.scan(operation_on_slice, carry, slices)
    
    return results

# Beispielarray und Slicelänge
array = jnp.arange(20)
slice_size = 5

# Wende die Schleife auf das Array an
processed_array = process_slices_scan(array, slice_size)
print(processed_array)