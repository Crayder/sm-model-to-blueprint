import numpy as np
from scipy.spatial import KDTree

# Mapping of block names to UUIDs
BLOCK_IDS = {
    "scrapwood": "1fc74a28-addb-451a-878d-c3c605d63811",
    "wood1": "df953d9c-234f-4ac2-af5e-f0490b223e71",
    "wood2": "1897ee42-0291-43e4-9645-8c5a5d310398",
    "wood3": "061b5d4b-0a6a-4212-b0ae-9e9681f1cbfb",
    "scrapmetal": "1f7ac0bb-ad45-4246-9817-59bdf7f7ab39",
    "metal1": "8aedf6c2-94e1-4506-89d4-a0227c552f1e",
    "metal2": "1016cafc-9f6b-40c9-8713-9019d399783f",
    "metal3": "c0dfdea5-a39d-433a-b94a-299345a5df46",
    "scrapstone": "30a2288b-e88e-4a92-a916-1edbfc2b2dac",
    "concrete1": "a6c6ce30-dd47-4587-b475-085d55c6a3b4",
    "concrete2": "ff234e42-5da4-43cc-8893-940547c97882",
    "concrete3": "e281599c-2343-4c86-886e-b2c1444e8810",
    "cardboard": "f0cba95b-2dc4-4492-8fd9-36546a4cb5aa",
    "sand": "c56700d9-bbe5-4b17-95ed-cef05bd8be1b",
    "plastic": "628b2d61-5ceb-43e9-8334-a4135566df7a",
    "glass": "5f41af56-df4c-4837-9b3c-10781335757f",
    "glasstile": "749f69e0-56c9-488c-adf6-66c58531818f",
    "armoredglass": "b5ee5539-75a2-4fef-873b-ef7c9398b3f5",
    "bubblewrap": "f406bf6e-9fd5-4aa0-97c1-0b3c2118198e",
    "restroom": "920b40c8-6dfc-42e7-84e1-d7e7e73128f6",
    "tiles": "8ca49bff-eeef-4b43-abd0-b527a567f1b7",
    "bricks": "0603b36e-0bdb-4828-b90c-ff19abcdfe34",
    "lights": "073f92af-f37e-4aff-96b3-d66284d5081c",
    "caution": "09ca2713-28ee-4119-9622-e85490034758",
    "crackedconcrete": "f5ceb7e3-5576-41d2-82d2-29860cf6e20e",
    "concretetiles": "cd0eff89-b693-40ee-bd4c-3500b23df44e",
    "metalbricks": "220b201e-aa40-4995-96c8-e6007af160de",
    "beam": "25a5ffe7-11b1-4d3e-8d7a-48129cbaf05e",
    "insulation": "9be6047c-3d44-44db-b4b9-9bcf8a9aab20",
    "drywall": "b145d9ae-4966-4af6-9497-8fca33f9aee3",
    "carpet": "febce8a6-6c05-4e5d-803b-dfa930286944",
    "plasticwall": "e981c337-1c8a-449c-8602-1dd990cbba3a",
    "metalnet": "4aa2a6f0-65a4-42e3-bf96-7dec62570e0b",
    "crossnet": "3d0b7a6e-5b40-474c-bbaf-efaa54890e6a",
    "tryponet": "ea6864db-bb4f-4a89-b9ec-977849b6713a",
    "stripednet": "a479066d-4b03-46b5-8437-e99fec3f43ee",
    "squarenet": "b4fa180c-2111-4339-b6fd-aed900b57093",
    "spaceshipmetal": "027bd4ec-b16d-47d2-8756-e18dc2af3eb6",
    "spaceshipfloor": "4ad97d49-c8a5-47f3-ace3-d56ba3affe50",
    "treadplate": "f7d4bfed-1093-49b9-be32-394c872a1ef4",
    "warehousefloor": "3e3242e4-1791-4f70-8d1d-0ae9ba3ee94c",
    "wornmetal": "d740a27d-cc0f-4866-9e07-6a5c516ad719",
    "framework": "c4a2ffa8-c245-41fb-9496-966c6ee4648b",
    "challenge01": "491b1d4f-3a00-403e-b64f-f9eb7bda7683",
    "challenge02": "cad3a585-2686-40e2-8eb1-02f5df20a021",
    "challengeglass": "17baf3ba-0b40-4eef-9823-119059d5c12d"
}

# Scrap Mechanic colors
SM_COLORS = [
    "EEEEEE", "7F7F7F", "4A4A4A", "222222", "F5F071", "E2DB13", "817C00", "323000",
    "CBF66F", "A0EA00", "577D07", "375000", "064023", "0E8031", "19E753", "68FF88",
    "7EEDED", "2CE6E6", "118787", "0F2E91", "0A1D5A", "0A4444", "0A3EE2", "4C6FE3",
    "AE79F0", "7514ED", "500AA6", "35086C", "472800", "520653", "560202", "673B00",
    "DF7F00", "EEAF5C", "EE7BF0", "F06767", "CF11D2", "720A74", "7C0000", "D02525"
]
SM_COLORS_RGB = [np.array([int(c[i:i+2], 16) / 255 for i in range(0, 6, 2)]) for c in SM_COLORS]
SM_COLORS_KDTREE = KDTree(SM_COLORS_RGB)

# Default values for the form inputs
CONFIG_VALUES = {
    "input_file": "testStuff.obj",
    "output_file": "C:\\Users\\Corey\\AppData\\Roaming\\Axolot Games\\Scrap Mechanic\\User\\User_76561198805744844\\Blueprints\\51c6485e-c45d-47b0-a3c9-08d2db23aef9\\blueprint.json",
    "voxel_scale": 1.0,
    "obj_scale": 3.0,
    "obj_offset": np.array([0.5, 0.5, 0.5]),
    "rotate_axis": 'x',
    "rotate_angle": 90,
    "use_set_color": False,
    "set_color": [0.0, 0.588235, 1.0],
    "use_set_block": False,
    "set_block": "wood1",
    "use_scrap_colors": True,
    "vary_colors": False,
    "interior_fill": False
}

# Default material based on default values
FALLBACK_MATERIAL = {
    "color": CONFIG_VALUES["set_color"],
    "shapeId": BLOCK_IDS[CONFIG_VALUES["set_block"]]
}
