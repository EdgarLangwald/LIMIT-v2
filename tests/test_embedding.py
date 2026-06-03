import numpy as np
from dataset_creation import build_dataset  # TODO: build_dataset not yet implemented in limit.dataset
from src.embed import embed_dataset


test_embedding_metadata()
# Does the number of embeddings equal number docs + number queries?
# Embedding dimension same as reported by model?


def test_embedding_statistics():
    dataset, _ = build_dataset(n=20, m=1, seed=42)
    result = embed_dataset(dataset, model_name="default")
    embs = np.concatenate([result["doc_embs"], result["qry_embs"]], axis=0)

    assert not np.isnan(embs).any()
    assert not np.isinf(embs).any()

    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Norms not unit: min={norms.min():.4f} max={norms.max():.4f}"

    mean_cos_sim = (embs @ embs.T).mean()  # diagonal included but won't affect much for large n
    assert mean_cos_sim < 0.95, f"Embeddings may be collapsed: mean cosine sim={mean_cos_sim:.4f}"

    sv = np.linalg.svd(embs, compute_uv=False)
    participation_ratio = sv.sum() ** 2 / (sv ** 2).sum()
    assert participation_ratio > 2.0, f"Embeddings may be dimension-collapsed: PR={participation_ratio:.2f}"


def test_embedding_STS():
    GROUPS = [
        {
            "similar": [
                "I grind fresh coffee beans every morning before brewing.",
                "The espresso machine must heat up before pulling a shot.",
                "Pour-over coffee needs a slow, steady pour to extract properly.",
            ],
            "unrelated": [
                "Hydrothermal vents on the ocean floor support unique ecosystems.",
                "Tax deductions for home offices must be calculated carefully.",
                "Knitting a cable pattern requires crossing stitches with a needle.",
            ],
        },
        {
            "similar": [
                "Teaching a dog to sit requires consistent reward timing.",
                "Positive reinforcement works better than punishment for training dogs.",
                "A dog that pulls on the leash needs to learn loose-leash walking.",
            ],
            "unrelated": [
                "Silicon wafers are etched with chemicals during chip manufacturing.",
                "Monet applied thick dabs of paint to capture light on water.",
                "Glaciers move slowly downhill under their own immense weight.",
            ],
        },
        {
            "similar": [
                "Altitude sickness can set in above 8,000 feet if you ascend too fast.",
                "Trekking poles reduce strain on your knees when hiking downhill.",
                "Always check trail conditions before setting out on a mountain hike.",
            ],
            "unrelated": [
                "Refinancing a mortgage can lower monthly payments when rates drop.",
                "Jazz musicians improvise solos over a chord progression called the changes.",
                "Parasites can complete their life cycles across multiple host species.",
            ],
        },
        {
            "similar": [
                "Al dente pasta should still have a slight bite when you taste it.",
                "A proper Italian ragù simmers for several hours to develop flavor.",
                "Fresh basil should be torn, not cut, to preserve its aroma.",
            ],
            "unrelated": [
                "Quantum entanglement links particles so that measuring one affects the other.",
                "Medieval trebuchets could hurl stone projectiles over castle walls.",
                "An optometrist checks for refractive errors like myopia and astigmatism.",
            ],
        },
        {
            "similar": [
                "Changing the oil every 5,000 miles keeps the engine running smoothly.",
                "Tire pressure should be checked monthly and before long road trips.",
                "Brake pads wear down and must be replaced before metal contacts metal.",
            ],
            "unrelated": [
                "Ballet dancers train for years to execute a perfect arabesque.",
                "Coral polyps secrete calcium carbonate to build reef structures.",
                "Sanskrit has one of the most complex grammatical systems of any language.",
            ],
        },
        {
            "similar": [
                "Setting a breakpoint lets you pause execution and inspect variable values.",
                "Reading the stack trace helps identify where an exception was thrown.",
                "Explaining your code aloud helps spot logic errors, a technique called rubber duck debugging.",
            ],
            "unrelated": [
                "Beehive frames must be inspected regularly to monitor the queen's health.",
                "Tectonic plates move a few centimeters per year due to convection currents.",
                "Haute couture garments are hand-sewn and fitted to individual clients.",
            ],
        },
        {
            "similar": [
                "Swimming in open water requires awareness of currents and tides.",
                "Wearing a wetsuit keeps body heat in when swimming in cold ocean water.",
                "Sighting by lifting your head to navigate is a key open-water swimming skill.",
            ],
            "unrelated": [
                "In chess, controlling the center early gives positional advantage.",
                "Patent applications for pharmaceuticals must demonstrate clinical efficacy.",
                "Ukrainian embroidery uses geometric patterns passed down through generations.",
            ],
        },
        {
            "similar": [
                "Getting lost in a novel for hours is one of the great pleasures of reading.",
                "A well-drawn character makes you forget they are fictional.",
                "Rereading a book often reveals details you missed on the first pass.",
            ],
            "unrelated": [
                "Lava flows can travel at speeds of several kilometers per hour.",
                "Foreign exchange rates fluctuate based on interest rates and trade balances.",
                "Origami cranes are folded from a single square of paper without cutting.",
            ],
        },
        {
            "similar": [
                "Holding warrior pose builds strength in the legs and core.",
                "Breathing consciously through each yoga pose deepens the practice.",
                "A regular yoga routine can improve both flexibility and mental calm.",
            ],
            "unrelated": [
                "Blast furnaces use coke and limestone to smelt iron ore into pig iron.",
                "Byzantine churches feature elaborate mosaic decoration on their domes.",
                "Cover crops like clover are planted between seasons to restore soil nitrogen.",
            ],
        },
        {
            "similar": [
                "A field guide helps identify birds by plumage, size, and behavior.",
                "Early morning is the best time to spot birds during peak singing hours.",
                "Binoculars with 8x magnification are ideal for most birdwatching.",
            ],
            "unrelated": [
                "Public-key cryptography relies on the difficulty of factoring large primes.",
                "Diesel engines use compression ignition rather than spark plugs.",
                "Ikebana, Japanese flower arrangement, emphasizes minimalism and asymmetry.",
            ],
        },
        {
            "similar": [
                "Calluses on your fingertips develop after weeks of regular guitar practice.",
                "Learning barre chords opens up new chord shapes across the fretboard.",
                "Fingerpicking requires independent control of each finger on the picking hand.",
            ],
            "unrelated": [
                "Immigration quotas and visa categories are set by federal statute.",
                "Stalactites form when mineral-laden water drips from cave ceilings.",
                "Silk production begins when silkworms spin cocoons of continuous fiber.",
            ],
        },
        {
            "similar": [
                "Deadheading spent flowers encourages plants to produce more blooms.",
                "Composting kitchen scraps returns nutrients to garden soil.",
                "Mulching around plants retains moisture and suppresses weeds.",
            ],
            "unrelated": [
                "Black holes warp spacetime so severely that not even light escapes.",
                "Competitive rowers synchronize their strokes precisely with the coxswain's calls.",
                "Gregorian chant uses a single melodic line without harmonic accompaniment.",
            ],
        },
        {
            "similar": [
                "A falling barometer often signals an approaching low-pressure system.",
                "Doppler radar reveals the internal wind structure of thunderstorms.",
                "Meteorologists use ensemble models to quantify forecast uncertainty.",
            ],
            "unrelated": [
                "Aged cheese develops complex flavor through enzymatic breakdown of proteins.",
                "Roman aqueducts carried water across valleys using precise gradient calculations.",
                "Crochet uses a single hook to create interlocked fabric loops.",
            ],
        },
        {
            "similar": [
                "The exposure triangle balances ISO, aperture, and shutter speed.",
                "Golden hour light just after sunrise gives photos a warm, soft quality.",
                "The rule of thirds guides where to place the subject in a photograph.",
            ],
            "unrelated": [
                "Anglerfish lure prey with a bioluminescent appendage in the deep ocean.",
                "Labor unions negotiate collective bargaining agreements with employers.",
                "Tibetan monks recite mantras while rotating prayer wheels.",
            ],
        },
        {
            "similar": [
                "A proper running warm-up includes dynamic stretches and an easy jog.",
                "Increasing weekly mileage too fast leads to overuse injuries like shin splints.",
                "Running economy improves with a high cadence and upright posture.",
            ],
            "unrelated": [
                "Glassblowers gather molten glass on a steel blowpipe and inflate it by lung.",
                "The Ottoman Empire controlled trade routes between Europe and Asia for centuries.",
                "Protein folding determines the three-dimensional structure and function of enzymes.",
            ],
        },
        {
            "similar": [
                "Tannins in red wine create a drying sensation on the gums and cheeks.",
                "Swirling wine in the glass releases volatile aromatic compounds.",
                "A wine's finish describes how long the flavors linger after swallowing.",
            ],
            "unrelated": [
                "Avalanches release when a weak layer in the snowpack collapses under stress.",
                "Typeface designers balance stroke width, spacing, and letterform proportion.",
                "Wetland conservation protects biodiversity and filters agricultural runoff.",
            ],
        },
        {
            "similar": [
                "Controlling the center with pawns and pieces is a key opening principle in chess.",
                "A pin immobilizes a piece because moving it would expose a more valuable piece.",
                "Endgame technique involves converting small advantages into a winning position.",
            ],
            "unrelated": [
                "Coral bleaching occurs when thermal stress causes corals to expel their algae.",
                "Perfumers blend top, heart, and base notes to create a lasting fragrance.",
                "Freight logistics coordinates warehousing, routing, and customs clearance.",
            ],
        },
        {
            "similar": [
                "Measuring twice before cutting prevents costly mistakes in carpentry.",
                "Priming drywall before painting prevents uneven absorption of finish coats.",
                "Tile grout should be sealed after installation to resist moisture and staining.",
            ],
            "unrelated": [
                "Tidal bulges are caused by the gravitational pull of the moon on the oceans.",
                "Neurosurgeons must avoid damaging eloquent cortex during tumor resection.",
                "Puppet theater traditions in Indonesia date back over a thousand years.",
            ],
        },
        {
            "similar": [
                "Dark skies far from city light pollution reveal thousands more stars.",
                "A telescope's aperture determines how much light it can gather from distant objects.",
                "Tracking mounts compensate for Earth's rotation to keep stars centered.",
            ],
            "unrelated": [
                "Tanning leather involves treating hide with tannin to prevent decomposition.",
                "Polka, a folk dance originating in Bohemia, is danced in fast duple time.",
                "Drip irrigation delivers water directly to plant roots, reducing evaporation.",
            ],
        },
        {
            "similar": [
                "Focusing attention on the breath is the foundation of many meditation practices.",
                "Mind wandering during meditation is normal; the practice is in returning attention.",
                "Regular meditation has been shown to reduce cortisol levels and perceived stress.",
            ],
            "unrelated": [
                "Cable-stayed bridges transfer deck load through diagonal cables to tall pylons.",
                "Taxidermists use wire armatures and foam forms to preserve animal specimens.",
                "Arctic explorers manage frostbite risk by layering moisture-wicking fabrics.",
            ],
        },
    ]

    all_sentences = [s for g in GROUPS for s in g["similar"] + g["unrelated"]]
    corpus = {str(i): s for i, s in enumerate(all_sentences)}
    result = embed_dataset({"corpus": corpus, "queries": {}}, model_name="default")
    embs = result["doc_embs"]  # (120, dim), L2-normalized

    for group_idx, _ in enumerate(GROUPS):
        base = group_idx * 6
        sim_embs = embs[base : base + 3]
        unrel_embs = embs[base + 3 : base + 6]

        sim_sim = sim_embs @ sim_embs.T      # [i,j] = cos(similar[i], similar[j])
        sim_unrel = sim_embs @ unrel_embs.T  # [i,j] = cos(similar[i], unrelated[j])
        unrel_unrel = unrel_embs @ unrel_embs.T

        for i in range(3):
            min_within = min(sim_sim[i, j] for j in range(3) if j != i)
            max_cross = sim_unrel[i].max()
            assert min_within > max_cross, (
                f"Group {group_idx}: similar[{i}] not closer to similar sentences than to unrelated "
                f"(min_within={min_within:.3f}, max_cross={max_cross:.3f})"
            )

        for i in range(3):
            mean_to_sim = sim_unrel[:, i].mean()
            mean_to_other_unrel = np.array([unrel_unrel[i, j] for j in range(3) if j != i]).mean()
            assert abs(mean_to_sim - mean_to_other_unrel) < 0.2, (
                f"Group {group_idx}: unrelated[{i}] has unusual affinity to similar cluster "
                f"(mean_to_sim={mean_to_sim:.3f}, mean_to_other_unrel={mean_to_other_unrel:.3f})"
            )

