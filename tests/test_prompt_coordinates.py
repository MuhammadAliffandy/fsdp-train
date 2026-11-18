from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_prompt_coordinates import position_contract, PromptOrder


class LinearFeatures:
    def forward_with_coords(self, coordinates, image_size):
        return coordinates / torch.tensor(image_size)


class Grid:
    input_image_size = (128, 128, 128)
    pe_layer = LinearFeatures()

    def __init__(self, swap):
        self.swap = swap

    def get_dense_pe(self):
        axes = torch.meshgrid(*[(torch.arange(8) + .5) / 8] * 3, indexing='ij')
        values = torch.stack(axes)
        return values[[1, 0, 2] if self.swap else [0, 1, 2]].unsqueeze(0)


class CoordinateTests(unittest.TestCase):
    def test_contract_distinguishes_native_and_swapped_grids(self):
        normal = position_contract(Grid(False))
        self.assertTrue(normal['native']['aligned_at_cell_centers'])
        self.assertFalse(normal['swap_xy']['aligned_at_cell_centers'])
        swapped = position_contract(Grid(True))
        self.assertFalse(swapped['native']['aligned_at_cell_centers'])
        self.assertTrue(swapped['swap_xy']['aligned_at_cell_centers'])

    def test_proxy_changes_only_prompt_coordinates(self):
        class Spy:
            def __call__(self, **kwargs):
                self.arguments = kwargs
            def get_dense_pe(self):
                return 'dense unchanged'
        spy = Spy()
        proxy = PromptOrder(spy, [1, 0, 2])
        point = torch.tensor([[[12., 56., 90.]]])
        label = torch.tensor([[1]])
        masks = torch.zeros((1, 1, 32, 32, 32))
        proxy(points=(point, label), boxes=None, masks=masks)
        torch.testing.assert_close(spy.arguments['points'][0], torch.tensor([[[56., 12., 90.]]]))
        torch.testing.assert_close(point, torch.tensor([[[12., 56., 90.]]]))
        self.assertIs(spy.arguments['points'][1], label)
        self.assertIs(spy.arguments['masks'], masks)
        self.assertEqual(proxy.get_dense_pe(), 'dense unchanged')
        self.assertEqual(len(proxy.trace), 1)


if __name__ == '__main__':
    unittest.main()
