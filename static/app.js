/* ==============================================================================
   AudioTag AI — Frontend Application Logic
   Dynamic Probability Sorting, Color Intensity & Multi-Layer Acoustic Dossier
   with Native Web Audio Timbre Synthesizer
   ============================================================================== */

let currentAudioFile = null;
let currentResults = null;
let currentSortMode = 'alpha'; // 'alpha' or 'prob'
let currentModalInstrument = null;
let audioCtx = null;

// ==============================================================================
// COMPREHENSIVE INSTRUMENT ACOUSTIC INTELLIGENCE DATABASE
// ==============================================================================
const INSTRUMENT_INTEL = {
  accordion: {
    name: "Accordion",
    family: "Aerophone (Free-Reed)",
    frequency: "30 Hz – 4,000 Hz",
    attack: "Bellows Breath Swell",
    feel: "Reedy, Shimmering & Nostalgic",
    dailyLifeSound: "An antique leather hearth bellows sighing open and closed, or a singing miniature steam engine powered by keyboard valves.",
    spotInSong: "Listen for a rich, vibrating reed shimmer that gently pulses and breathes in the background, often driving a bouncy 1-2-3 waltz cadence behind the lead melody.",
    famousCue: "The nostalgic, romantic Parisian soundtrack of the movie 'Amélie' (Yann Tiersen), or traditional Latin American cumbia and polka accordions.",
    timbre: "Dual detuned free-reeds creating signature musette vibrato beating",
    origin: "19th century Europe (patented in Berlin & Vienna, 1820s)",
    famousContext: "French musette, Argentine tango, Balkan folk, Tex-Mex norteño, polka."
  },
  bass: {
    name: "Bass",
    family: "Chordophone (Low-Frequency String)",
    frequency: "41 Hz – 300 Hz",
    attack: "Heavy Plucked Punch",
    feel: "Subterranean Rumble & Chest Pressure",
    dailyLifeSound: "A heavy diesel truck engine idling at a red light that vibrates your rear-view mirror, or distant summer thunder rolling across hills.",
    spotInSong: "Don't just listen with your ears—feel it in your ribcage. It locks in directly with the kick drum to give music its physical groove and harmonic floor.",
    famousCue: "The menacing, infectious low-end groove in Billie Eilish's 'Bad Guy', or the legendary opening bassline in Queen's 'Another One Bites the Dust'.",
    timbre: "Sub-heavy, deep, punchy fundamental anchoring the entire mix",
    origin: "Evolved from the orchestral double bass to Leo Fender's electric Precision Bass in 1951",
    famousContext: "Funk basslines, jazz walking bass, reggae dub grooves, rock power foundations."
  },
  cello: {
    name: "Cello",
    family: "Bowed String (Chordophone)",
    frequency: "65 Hz – 1,000 Hz",
    attack: "Smooth Bow Friction",
    feel: "Velvety, Dark & Mournful",
    dailyLifeSound: "A deep, soulful human voice speaking in a quiet wood-paneled room; dark, resonant, comforting, and intensely romantic.",
    spotInSong: "Listen for warm, sweeping low-string sustained notes that sit just above the bass, bringing deep emotional warmth to film scores and acoustic ballads.",
    famousCue: "Bach's Cello Suite No. 1 in G Major (the world's most famous cello piece), or the weeping strings in the 'Game of Thrones' main theme.",
    timbre: "Velvety, dark, resonant, rich second and third harmonics",
    origin: "Northern Italy in the early 16th century (Amati & Stradivari families)",
    famousContext: "Bach suites, orchestral film scores, string quartets, acoustic ballads."
  },
  clarinet: {
    name: "Clarinet",
    family: "Woodwind (Single-Reed Aerophone)",
    frequency: "105 Hz – 2,000 Hz",
    attack: "Fluid Woody Swell",
    feel: "Round, Hollow & Warm",
    dailyLifeSound: "Warm spiced honey poured through carved hollow aged wood, or a mellow rounded wooden windpipe singing in a quiet forest.",
    spotInSong: "It sounds smoother than a saxophone, without the metallic buzz. Listen for fluid, rapid woody runs with a round, warm low register (the chalumeau).",
    famousCue: "The legendary, seductive rising glissando intro to George Gershwin's 'Rhapsody in Blue', or Benny Goodman swing jazz.",
    timbre: "Cylindrical bore producing strictly odd-harmonic overtones",
    origin: "Germany (Johann Christoph Denner in Nuremberg, circa 1700)",
    famousContext: "Mozart clarinet concerto, swing jazz big bands, klezmer dances."
  },
  cymbals: {
    name: "Cymbals",
    family: "Idiophone (Struck Bronze Metal)",
    frequency: "3,000 Hz – 18,000 Hz",
    attack: "Instant Metallic Shatter",
    feel: "Explosive Sizzle & High Shimmer",
    dailyLifeSound: "Water droplets splashing and sizzling violently on a blistering hot cast-iron skillet, or the explosive crash of heavy bronze shields.",
    spotInSong: "Look for the explosive burst of high-frequency white sizzle that marks song transitions (crashes), or the steady ticking pulse keeping time (hi-hat and ride).",
    famousCue: "The explosive crash on beat 1 of Nirvana's 'Smells Like Teen Spirit', or the delicate jazz ride cymbal pulse in Dave Brubeck's 'Take Five'.",
    timbre: "Complex inharmonic metallic partials and high-frequency white noise shimmer",
    origin: "Bronze Age Anatolia (modern Turkey / Zildjian tradition dating back to 1623)",
    famousContext: "Drum kit crash accents, ride cymbal jazz pulse, symphonic climaxes."
  },
  drums: {
    name: "Drums",
    family: "Membranophone (Struck Acoustic Skin)",
    frequency: "50 Hz – 8,000 Hz",
    attack: "Sharp Percussive Transient",
    feel: "Visceral Impact & Driving Pulse",
    dailyLifeSound: "Heavy boots marching in cadence on hard packed earth, a racing heartbeat, or a heavy basketball bouncing violently on hardwood.",
    spotInSong: "The foundational pulse of the track—listen for the low 'thump' of the kick drum paired with the sharp 'crack' of the snare on beats 2 and 4.",
    famousCue: "The colossal stadium drum beat in Queen's 'We Will Rock You', or the punchy groove in Led Zeppelin's 'When the Levee Breaks'.",
    timbre: "Punchy percussive attack, rapid decay, resonant drum shell resonance",
    origin: "Prehistoric worldwide origins; modern drum kit assembled in early 20th century USA",
    famousContext: "The rhythmic backbone across rock, hip-hop, funk, pop, and Afrobeat."
  },
  flute: {
    name: "Flute",
    family: "Woodwind (Edge-Blown Aerophone)",
    frequency: "260 Hz – 2,500 Hz",
    attack: "Airy Breath Flutter",
    feel: "Pure, Silky & Crystalline",
    dailyLifeSound: "Blowing air gently across the open rim of a glass Coca-Cola bottle, or the sweet high trill of songbirds echoing through mountain pines.",
    spotInSong: "Listen for a breathy, hollow flutter dancing effortlessly above the vocals and guitars during quiet breakdowns, acoustic bridges, or cinematic intros.",
    famousCue: "The iconic, heart-melting whistle-like intro to Celine Dion's 'My Heart Will Go On' (Titanic), or The Mamas & the Papas' 'California Dreamin'.",
    timbre: "Near-sinusoidal pure fundamental with gentle turbulent breath noise",
    origin: "Prehistoric bone flutes (>40,000 years old); modern silver Boehm key system developed in Germany (1847)",
    famousContext: "Classical orchestra solos, Celtic traditional folk, Andean panpipe tunes, jazz flute."
  },
  guitar: {
    name: "Guitar",
    family: "Plucked String (Chordophone)",
    frequency: "80 Hz – 1,200 Hz",
    attack: "Crisp Plucked Transient",
    feel: "Twangy, Resonant & Dynamic",
    dailyLifeSound: "Snapping taut nylon or steel fishing wire against a hollow cedar cigar box, or rain tapping briskly on a wooden porch.",
    spotInSong: "Listen for the rhythmic strumming of full harmonic chords or sharp single-note melodic lines with metallic twang and natural acoustic body decay.",
    famousCue: "The acoustic fingerpicking in Led Zeppelin's 'Stairway to Heaven', or the overdriven power riff opening AC/DC's 'Back in Black'.",
    timbre: "Sharp plucked transient decaying into rich harmonic body overtones",
    origin: "Derived from the Spanish vihuela and Renaissance lute in 16th-century Spain",
    famousContext: "Acoustic fingerstyle, rock guitar overdrive, flamenco rasgueados, indie pop."
  },
  mallet_percussion: {
    name: "Mallet Percussion",
    family: "Tuned Percussion (Idiophone)",
    frequency: "100 Hz – 4,000 Hz",
    attack: "Sharp Glassy / Woody Strike",
    feel: "Crystalline Chime & Bell Bloom",
    dailyLifeSound: "Heavy raindrops striking crystal wine glasses, a music box playing on a nightstand, or hollow wooden blocks knocking in a courtyard.",
    spotInSong: "Listen for bell-like, crystalline pings or woody, hollow taps that sustain with shimmering, glassy overtones over the rhythm section.",
    famousCue: "The vibraphone opening in The Police's 'Every Little Thing She Does Is Magic', or the magical chime melodies in Harry Potter's 'Hedwig's Theme'.",
    timbre: "Inharmonic struck bar partials tuned to clean acoustic octaves and tenths",
    origin: "Southeast Asia and Africa (Xylophones/Balafons), refined in 19th-century classical orchestras",
    famousContext: "Vibraphone jazz solos, orchestral marimba concertos, glockenspiel fairy-tale motifs."
  },
  mandolin: {
    name: "Mandolin",
    family: "Plucked String (Lute Family)",
    frequency: "196 Hz – 1,800 Hz",
    attack: "Rapid Double-Pick Transient",
    feel: "Bright, Shimmering & Crisp",
    dailyLifeSound: "Sunlight dancing across a fast bubbling stone fountain—rapid, fluttering, metallic, cheerful, and crisp.",
    spotInSong: "Listen for rapid, shimmering paired-string strumming (tremolo) that sounds higher, tighter, and more percussive than a standard acoustic guitar.",
    famousCue: "The signature opening riff of R.E.M.'s 'Losing My Religion', or Led Zeppelin's 'The Battle of Evermore'.",
    timbre: "Paired unison metal strings creating a rich acoustic chorusing shimmer",
    origin: "18th century Italy (evolved from the Renaissance mandora in Naples)",
    famousContext: "Bluegrass breakdowns, Italian Neapolitan folk songs, Celtic reels."
  },
  piano: {
    name: "Piano",
    family: "Keyboard (Hammer-Struck Chordophone)",
    frequency: "27.5 Hz – 4,186 Hz",
    attack: "Felt Hammer Percussive Strike",
    feel: "Expansive, Resonant & Dynamic",
    dailyLifeSound: "Dense felt hammers striking tightly coiled brass wires, echoing like pure water droplets falling into a deep, still mountain lake.",
    spotInSong: "The king of acoustic versatility—listen for crisp, percussive chord attacks that sustain into rich, reverberant room resonance across low and high registers.",
    famousCue: "The melancholic, haunting opening of Adele's 'Someone Like You', or the driving hypnotic chords of Coldplay's 'Clocks'.",
    timbre: "Percussive strike transient decaying across rich harmonic string partials",
    origin: "Invented by Bartolomeo Cristofori in Florence, Italy (around 1700)",
    famousContext: "Chopin nocturnes, Beethoven sonatas, jazz improvisation, iconic pop ballads."
  },
  saxophone: {
    name: "Saxophone",
    family: "Single-Reed Woodwind / Brass Hybrid",
    frequency: "140 Hz – 1,000 Hz",
    attack: "Vocalized Reed Bite",
    feel: "Smoky, Brassy & Sensual",
    dailyLifeSound: "Smoky midnight velvet, the brassy purr of a golden cat, or a passionate human cry channeled through glowing polished brass.",
    spotInSong: "Combines the expressive volume of brass with the reedy warmth of woodwinds. Listen for soulful, weeping solo melodies that slide smoothly between notes.",
    famousCue: "The unforgettable, sultry opening saxophone line in George Michael's 'Careless Whisper', or Gerry Rafferty's 'Baker Street'.",
    timbre: "Conical brass bore driven by a reed, creating warm, vocal-like harmonic formants",
    origin: "Patented by Belgian instrument designer Adolphe Sax in Paris (1846)",
    famousContext: "John Coltrane jazz ballads, bebop solos, R&B horn sections, passionate soul music."
  },
  synthesizer: {
    name: "Synthesizer",
    family: "Electrophone (Electronic Sound Generator)",
    frequency: "20 Hz – 20,000 Hz",
    attack: "Instant Voltage Step or Swept Filter",
    feel: "Futuristic, Neon & Hypnotic",
    dailyLifeSound: "Futuristic laser beams cutting through night mist, buzzing neon light tubes, or the pulsing electronic hum of a 1980s retro arcade cabinet.",
    spotInSong: "Anything electronic, unnatural, and electric—listen for sweeping resonance filters, robotic arpeggios, pulsing synthwave basslines, or lush sci-fi ambient pads.",
    famousCue: "The driving 80s synth hook in The Weeknd's 'Blinding Lights', or the iconic retro theme of Netflix's 'Stranger Things'.",
    timbre: "Sawtooth, square, and pulse waveforms sculpted by resonant lowpass filters",
    origin: "Mid-20th century modular systems (Robert Moog, Don Buchla, and ARP in the 1960s)",
    famousContext: "Synthwave, 80s synthpop, ambient cinematic pads, EDM basslines, sci-fi soundtracks."
  },
  trombone: {
    name: "Trombone",
    family: "Brass (Lip-Vibrated Slide Aerophone)",
    frequency: "80 Hz – 500 Hz",
    attack: "Brassy Lip Slap & Slide",
    feel: "Bold, Gliding & Heroic",
    dailyLifeSound: "A giant sliding brass siren, or the deep, comical, majestic horn of a steam barge gliding down a wide river.",
    spotInSong: "Listen for smooth, continuous pitch-slides (glissando) and deep, bold brass blasts that add punch to Latin, jazz, and movie scores.",
    famousCue: "Big band brass stabs, the Imperial March brass in Star Wars, or trombone solos in classic ska and salsa tracks.",
    timbre: "Cylindrical brass tubing creating bright, cutting, and majestic brass overtones",
    origin: "The Renaissance sackbut in 15th-century Europe",
    famousContext: "Big band jazz riffs, salsa brass sections, symphonic heroic fanfares."
  },
  trumpet: {
    name: "Trumpet",
    family: "Brass (Valved Aerophone)",
    frequency: "160 Hz – 1,000 Hz",
    attack: "Sharp Crisp Fanfare",
    feel: "Piercing, Regal & Brilliant",
    dailyLifeSound: "A piercing golden sunbeam cutting through thunderclouds, a royal castle coronation fanfare, or a crisp military bugle at dawn.",
    spotInSong: "The brightest, sharpest instrument in the horn section—cuts directly through the thickest audio mix with regal clarity and high-register power.",
    famousCue: "Miles Davis in 'So What', the triumphant fanfare in the Rocky theme 'Gonna Fly Now', or mariachi brass lines.",
    timbre: "High harmonic density with brilliant, sharp brassy overtones",
    origin: "Ancient signaling horns, modernized with piston valves in 1818 Germany",
    famousContext: "Miles Davis cool jazz, military bugle calls, mariachi brass, symphonic climaxes."
  },
  ukulele: {
    name: "Ukulele",
    family: "Plucked String (Chordophone)",
    frequency: "260 Hz – 1,000 Hz",
    attack: "Quick Strum Snap",
    feel: "Playful, Breezy & Cheerful",
    dailyLifeSound: "Strumming sunshine on a warm wooden beach porch; lightweight, plucky, carefree, and breezy.",
    spotInSong: "Sounds like a small, high-pitched acoustic guitar with a cheerful, dry nylon-string bounce and almost no low-end weight.",
    famousCue: "Israel Kamakawiwoʻole's timeless medley 'Somewhere Over the Rainbow / What a Wonderful World', or Jason Mraz's 'I'm Yours'.",
    timbre: "Light, dry, fast acoustic decay without heavy low-frequency resonance",
    origin: "Adapted by Portuguese immigrants in Hawaii (1880s) from the machete instrument",
    famousContext: "Hawaiian traditional music, indie acoustic pop covers, relaxed campfire songs."
  },
  violin: {
    name: "Violin",
    family: "Bowed String (Chordophone)",
    frequency: "196 Hz – 3,500 Hz",
    attack: "Sweet Bowed Swell",
    feel: "Soaring, Silky & Expressive",
    dailyLifeSound: "A sharp ribbon of silk slicing cleanly through crisp winter air, or a soaring soprano voice singing with intense emotion.",
    spotInSong: "The highest voice of the string family. Listen for rapid, agile melodic runs and sustained, crying vibrato that carries the main emotional melody.",
    famousCue: "The rapid, fiery strings in Vivaldi's 'Summer' (Four Seasons), or the weeping emotional theme in 'Schindler's List'.",
    timbre: "Brilliant, rich harmonic spectrum shaped by wooden bridge and spruce top plate",
    origin: "Cremona, Italy in the 16th century (Andrea Amati and Antonio Stradivari)",
    famousContext: "Vivaldi's Four Seasons, Appalachian fiddle breakdowns, cinematic emotional themes."
  },
  voice: {
    name: "Voice",
    family: "Vocal Signal (Human Vocal Tract)",
    frequency: "85 Hz – 1,100 Hz (Fundamental)",
    attack: "Organic Glottal Breath",
    feel: "Living, Articulate & Emotional",
    dailyLifeSound: "The most organic acoustic instrument in existence—conveying speech, vowels, whispers, hums, and emotional cries with living breath.",
    spotInSong: "The natural focal point of almost all vocal music—listen for lyrical formants, consonant articulation, and dynamic human vibrato.",
    famousCue: "Freddie Mercury's soaring vocals in 'Bohemian Rhapsody', or Whitney Houston's 'I Will Always Love You'.",
    timbre: "Continuously varying formants produced by tongue, pharynx, and lip resonance",
    origin: "The earliest musical instrument in human evolution (>100,000 years)",
    famousContext: "Lead vocals, choral harmonies, operatic arias, hip-hop flows, spoken word."
  }
};

// ==============================================================================
// NATIVE WEB AUDIO TIMBRE SYNTHESIZER
// ==============================================================================
function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

function playAcousticSignature(instKey) {
  const ctx = getAudioContext();
  const now = ctx.currentTime;
  const timbreBtn = document.getElementById('btn-play-timbre');
  const timbreBtnText = document.getElementById('timbre-btn-text');

  if (timbreBtn) {
    timbreBtn.classList.add('playing');
    if (timbreBtnText) timbreBtnText.textContent = 'Playing...';
    setTimeout(() => {
      timbreBtn.classList.remove('playing');
      if (timbreBtnText) timbreBtnText.textContent = 'Play Timbre';
    }, 1600);
  }

  // Master Gain for safety
  const master = ctx.createGain();
  master.gain.setValueAtTime(0.3, now);
  master.connect(ctx.destination);

  switch (instKey) {
    case 'flute': {
      // 587.33 Hz (D5) Sine + Soft Breath Noise + Vibrato
      const osc = ctx.createOscillator();
      const vibrato = ctx.createOscillator();
      const vibratoGain = ctx.createGain();
      const env = ctx.createGain();

      vibrato.frequency.value = 5.5; // 5.5 Hz vibrato
      vibratoGain.gain.value = 4.0;
      vibrato.connect(osc.frequency);

      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, now);

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.4, now + 0.12);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.4);

      osc.connect(env);
      env.connect(master);

      vibrato.start(now);
      osc.start(now);
      vibrato.stop(now + 1.4);
      osc.stop(now + 1.4);
      break;
    }

    case 'bass': {
      // 55 Hz (A1) Sub Sine + Triangle Punch
      const osc = ctx.createOscillator();
      const sub = ctx.createOscillator();
      const env = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(220, now);

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(55, now);
      sub.type = 'sine';
      sub.frequency.setValueAtTime(55, now);

      env.gain.setValueAtTime(0.01, now);
      env.gain.linearRampToValueAtTime(0.8, now + 0.04);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.2);

      osc.connect(filter);
      sub.connect(filter);
      filter.connect(env);
      env.connect(master);

      osc.start(now);
      sub.start(now);
      osc.stop(now + 1.2);
      sub.stop(now + 1.2);
      break;
    }

    case 'cello': {
      // 130.81 Hz (C3) Sawtooth with warm Formant Filters
      const osc = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(130.81, now);

      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(450, now);
      filter.Q.value = 2.5;

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.5, now + 0.18);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.5);

      osc.connect(filter);
      filter.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 1.5);
      break;
    }

    case 'clarinet': {
      // 349.23 Hz (F4) Square Wave (Pure Odd Harmonics)
      const osc = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      osc.type = 'square';
      osc.frequency.setValueAtTime(349.23, now);

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1100, now);

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.35, now + 0.08);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.3);

      osc.connect(filter);
      filter.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 1.3);
      break;
    }

    case 'cymbals': {
      // White noise burst + Highpass sizzle
      const bufferSize = ctx.sampleRate * 1.5;
      const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const output = noiseBuffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
      }

      const whiteNoise = ctx.createBufferSource();
      whiteNoise.buffer = noiseBuffer;

      const filter = ctx.createBiquadFilter();
      filter.type = 'highpass';
      filter.frequency.setValueAtTime(5000, now);

      const env = ctx.createGain();
      env.gain.setValueAtTime(0.7, now);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.5);

      whiteNoise.connect(filter);
      filter.connect(env);
      env.connect(master);

      whiteNoise.start(now);
      whiteNoise.stop(now + 1.5);
      break;
    }

    case 'drums': {
      // Kick drum pitch drop (140Hz -> 45Hz) + Snare noise snap
      const osc = ctx.createOscillator();
      const env = ctx.createGain();

      osc.frequency.setValueAtTime(140, now);
      osc.frequency.exponentialRampToValueAtTime(45, now + 0.15);

      env.gain.setValueAtTime(0.9, now);
      env.gain.exponentialRampToValueAtTime(0.001, now + 0.5);

      osc.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 0.5);
      break;
    }

    case 'guitar': {
      // 220 Hz Plucked String
      const osc = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(220, now);

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1800, now);
      filter.frequency.exponentialRampToValueAtTime(300, now + 0.8);

      env.gain.setValueAtTime(0.7, now);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.2);

      osc.connect(filter);
      filter.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 1.2);
      break;
    }

    case 'mallet_percussion': {
      // 880 Hz Crystalline Bell Chime
      const osc = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const env = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, now);
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(880 * 2.76, now); // Metallic overtone

      env.gain.setValueAtTime(0.6, now);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.4);

      osc.connect(env);
      osc2.connect(env);
      env.connect(master);

      osc.start(now);
      osc2.start(now);
      osc.stop(now + 1.4);
      osc2.stop(now + 1.4);
      break;
    }

    case 'mandolin': {
      // Rapid double-pluck tremolo at 659.25 Hz (E5)
      [0, 0.09, 0.18].forEach(delay => {
        const osc = ctx.createOscillator();
        const env = ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(659.25, now + delay);
        env.gain.setValueAtTime(0.4, now + delay);
        env.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.35);
        osc.connect(env);
        env.connect(master);
        osc.start(now + delay);
        osc.stop(now + delay + 0.35);
      });
      break;
    }

    case 'piano': {
      // 261.63 Hz (C4) Hammer strike + Rich decaying partials
      [261.63, 523.25, 784.88].forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const env = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, now);
        const amp = 0.5 / (idx + 1);
        env.gain.setValueAtTime(amp, now);
        env.gain.exponentialRampToValueAtTime(0.001, now + 1.5 - idx * 0.2);
        osc.connect(env);
        env.connect(master);
        osc.start(now);
        osc.stop(now + 1.5);
      });
      break;
    }

    case 'saxophone': {
      // 293.66 Hz (D4) Sawtooth + Vocal Formant Filter + Vibrato
      const osc = ctx.createOscillator();
      const vibrato = ctx.createOscillator();
      const vibratoGain = ctx.createGain();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      vibrato.frequency.value = 5.0;
      vibratoGain.gain.value = 6.0;
      vibrato.connect(osc.frequency);

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(293.66, now);

      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(950, now);
      filter.Q.value = 3.0;

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.5, now + 0.12);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.4);

      osc.connect(filter);
      filter.connect(env);
      env.connect(master);

      vibrato.start(now);
      osc.start(now);
      vibrato.stop(now + 1.4);
      osc.stop(now + 1.4);
      break;
    }

    case 'synthesizer': {
      // Detuned Dual Sawtooth Chord with Resonant Filter Sweep
      [261.63, 329.63, 392.00].forEach(f => {
        const osc = ctx.createOscillator();
        const filter = ctx.createBiquadFilter();
        const env = ctx.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(f, now);

        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(400, now);
        filter.frequency.exponentialRampToValueAtTime(3200, now + 0.4);
        filter.frequency.exponentialRampToValueAtTime(600, now + 1.4);
        filter.Q.value = 4.0;

        env.gain.setValueAtTime(0.25, now);
        env.gain.exponentialRampToValueAtTime(0.001, now + 1.4);

        osc.connect(filter);
        filter.connect(env);
        env.connect(master);

        osc.start(now);
        osc.stop(now + 1.4);
      });
      break;
    }

    case 'trombone': {
      // 110 Hz Brass Sawtooth with slight slide glissando
      const osc = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(110, now);
      osc.frequency.linearRampToValueAtTime(123.47, now + 0.5); // Glissando

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(600, now);

      env.gain.setValueAtTime(0.01, now);
      env.gain.linearRampToValueAtTime(0.6, now + 0.1);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.3);

      osc.connect(filter);
      filter.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 1.3);
      break;
    }

    case 'trumpet': {
      // 440 Hz Bright Brass Fanfare Swell
      const osc = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(440, now);

      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(1400, now);
      filter.Q.value = 2.0;

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.6, now + 0.07);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.3);

      osc.connect(filter);
      filter.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 1.3);
      break;
    }

    case 'ukulele': {
      // High-pitched cheerful nylon string pluck at 440 Hz
      const osc = ctx.createOscillator();
      const env = ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(440, now);

      env.gain.setValueAtTime(0.6, now);
      env.gain.exponentialRampToValueAtTime(0.001, now + 0.7);

      osc.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 0.7);
      break;
    }

    case 'violin': {
      // 659.25 Hz (E5) Sawtooth with rich 6Hz vibrato
      const osc = ctx.createOscillator();
      const vibrato = ctx.createOscillator();
      const vibratoGain = ctx.createGain();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      vibrato.frequency.value = 6.0;
      vibratoGain.gain.value = 8.0;
      vibrato.connect(osc.frequency);

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(659.25, now);

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(2800, now);

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.45, now + 0.15);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.4);

      osc.connect(filter);
      filter.connect(env);
      env.connect(master);

      vibrato.start(now);
      osc.start(now);
      vibrato.stop(now + 1.4);
      osc.stop(now + 1.4);
      break;
    }

    case 'voice': {
      // Formant synthesized human vowel ("Ahhh") at 220 Hz
      const osc = ctx.createOscillator();
      const f1 = ctx.createBiquadFilter();
      const f2 = ctx.createBiquadFilter();
      const env = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(220, now);

      f1.type = 'bandpass';
      f1.frequency.setValueAtTime(700, now);
      f1.Q.value = 4.0;

      f2.type = 'bandpass';
      f2.frequency.setValueAtTime(1220, now);
      f2.Q.value = 4.0;

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.5, now + 0.2);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.4);

      osc.connect(f1);
      osc.connect(f2);
      f1.connect(env);
      f2.connect(env);
      env.connect(master);

      osc.start(now);
      osc.stop(now + 1.4);
      break;
    }

    case 'accordion': {
      // Musette-tuned dual reeds (440 Hz + 442.5 Hz)
      const r1 = ctx.createOscillator();
      const r2 = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const env = ctx.createGain();

      r1.type = 'sawtooth';
      r1.frequency.setValueAtTime(440, now);
      r2.type = 'sawtooth';
      r2.frequency.setValueAtTime(442.5, now);

      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(1200, now);
      filter.Q.value = 1.5;

      env.gain.setValueAtTime(0.001, now);
      env.gain.linearRampToValueAtTime(0.4, now + 0.15);
      env.gain.exponentialRampToValueAtTime(0.001, now + 1.4);

      r1.connect(filter);
      r2.connect(filter);
      filter.connect(env);
      env.connect(master);

      r1.start(now);
      r2.start(now);
      r1.stop(now + 1.4);
      r2.stop(now + 1.4);
      break;
    }

    default:
      break;
  }
}

// ==============================================================================
// DOM EVENT HANDLERS
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('audio-input');
  const filePreview = document.getElementById('file-preview-card');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');
  const audioPlayer = document.getElementById('audio-player');
  const btnAnalyze = document.getElementById('btn-analyze');
  const resultsSection = document.getElementById('results-section');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdLabel = document.getElementById('threshold-label');
  const btnExportJson = document.getElementById('btn-export-json');

  // Taxonomy sorting buttons
  const sortAlphaBtn = document.getElementById('sort-alpha-btn');
  const sortProbBtn = document.getElementById('sort-prob-btn');

  // Modal elements
  const modal = document.getElementById('instrument-modal');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const btnPlayTimbre = document.getElementById('btn-play-timbre');

  // Click on dropzone triggers file picker
  dropzone.addEventListener('click', () => fileInput.click());

  // Drag and drop handlers
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    currentAudioFile = file;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

    const fileUrl = URL.createObjectURL(file);
    audioPlayer.src = fileUrl;
    filePreview.style.display = 'block';
  }

  // Analyze button click
  btnAnalyze.addEventListener('click', async () => {
    if (!currentAudioFile) return;

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `
      <span style="display:inline-block; animation: spin 1s linear infinite;">·</span>
      <span>Analyzing Acoustic Mix...</span>
    `;

    const formData = new FormData();
    formData.append('file', currentAudioFile);

    const thresholdVal = parseFloat(thresholdSlider.value);

    try {
      const response = await fetch(`/analyze?threshold=${thresholdVal}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Inference returned HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      currentResults = data;
      renderResults(data);

      // Enable and activate probability sort on benchmark grid
      if (sortProbBtn) {
        sortProbBtn.disabled = false;
        sortProbBtn.title = "Sort 18 instruments by latest detected probability";
      }

      resultsSection.style.display = 'block';
      resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
      alert(err.message || 'Error occurred during inference');
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = `
        <span>Analyze Track</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      `;
    }
  });

  // Dynamic slider adjustment (real-time threshold update)
  thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    thresholdLabel.textContent = val.toFixed(2);
    if (currentResults) {
      updateActiveThreshold(val);
    }
  });

  // Export JSON
  btnExportJson.addEventListener('click', () => {
    if (!currentResults) return;
    const blob = new Blob([JSON.stringify(currentResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentAudioFile ? currentAudioFile.name : 'audio'}_audiotag_results.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Taxonomy sorting buttons
  if (sortAlphaBtn && sortProbBtn) {
    sortAlphaBtn.addEventListener('click', () => {
      currentSortMode = 'alpha';
      sortAlphaBtn.classList.add('active');
      sortProbBtn.classList.remove('active');
      sortTaxonomyGrid('alpha');
    });

    sortProbBtn.addEventListener('click', () => {
      if (sortProbBtn.disabled) return;
      currentSortMode = 'prob';
      sortProbBtn.classList.add('active');
      sortAlphaBtn.classList.remove('active');
      sortTaxonomyGrid('prob');
    });
  }

  // Wire benchmark cards for modal clicks
  document.querySelectorAll('#taxonomy-grid .inst-card').forEach(card => {
    card.addEventListener('click', () => {
      const instKey = card.dataset.instrument;
      const score = (currentResults && currentResults.predictions) ? currentResults.predictions[instKey] : null;
      let rank = null;
      if (currentResults && currentResults.predictions) {
        const sorted = Object.entries(currentResults.predictions).sort((a, b) => b[1] - a[1]);
        const idx = sorted.findIndex(e => e[0] === instKey);
        if (idx !== -1) rank = idx + 1;
      }
      openInstrumentDossier(instKey, score, rank);
    });
  });

  // Timbre Player button
  if (btnPlayTimbre) {
    btnPlayTimbre.addEventListener('click', () => {
      if (currentModalInstrument) {
        playAcousticSignature(currentModalInstrument);
      }
    });
  }

  // Modal close handlers
  if (modalCloseBtn) {
    modalCloseBtn.addEventListener('click', closeInstrumentDossier);
  }

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeInstrumentDossier();
      }
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && modal.classList.contains('open')) {
      closeInstrumentDossier();
    }
  });
});

/**
 * Calculates dynamic color intensity based on instrument probability score.
 */
function getColorIntensity(score) {
  if (score >= 0.75) {
    return {
      bg: `rgba(255, 107, 0, ${0.08 + score * 0.08})`,
      border: `rgba(255, 107, 0, 0.45)`,
      barGradient: `linear-gradient(90deg, #ff6b00, #ff8c38)`,
      textColor: `#c44f00`,
      badgeStyle: `background: #fff3eb; color: #c44f00; border: 1px solid #ffd8c2;`
    };
  } else if (score >= 0.40) {
    return {
      bg: `rgba(224, 104, 16, ${0.04 + score * 0.06})`,
      border: `rgba(224, 104, 16, 0.35)`,
      barGradient: `linear-gradient(90deg, #e06810, #f8a055)`,
      textColor: `#a84500`,
      badgeStyle: `background: #fdf5ef; color: #a84500; border: 1px solid #fae1d0;`
    };
  } else if (score >= 0.15) {
    return {
      bg: `rgba(160, 150, 140, 0.04)`,
      border: `rgba(180, 175, 165, 0.4)`,
      barGradient: `linear-gradient(90deg, #999083, #b8b1a5)`,
      textColor: `#69716d`,
      badgeStyle: `background: #f3f4f1; color: #5f6964; border: 1px solid #e2e5df;`
    };
  } else {
    return {
      bg: `#fafbfa`,
      border: `var(--line)`,
      barGradient: `#dfe5df`,
      textColor: `#8a9490`,
      badgeStyle: `background: #f0f2ee; color: #8a9490; border: 1px solid #e5e8e3;`
    };
  }
}

/**
 * Renders the prediction results with dynamic sorting, ranks, and color intensity.
 */
function renderResults(data) {
  const container = document.getElementById('instruments-container');
  container.innerHTML = '';

  const predictions = data.predictions || {};
  const currentThreshold = parseFloat(document.getElementById('threshold-slider').value);

  // Sort descending by score
  const entries = Object.entries(predictions).sort((a, b) => b[1] - a[1]);

  let activeCount = 0;
  let topInstrument = entries.length > 0 ? entries[0] : null;

  entries.forEach(([inst, score], index) => {
    const isActive = score >= currentThreshold;
    if (isActive) activeCount++;

    const pct = Math.round(score * 100);
    const intel = INSTRUMENT_INTEL[inst] || { name: inst, family: 'Instrument' };
    const styleInfo = getColorIntensity(score);

    const card = document.createElement('div');
    card.className = `inst-card ${isActive ? 'active' : 'inactive'}`;
    card.dataset.score = score;
    card.dataset.instrument = inst;

    // Apply dynamic intensity styles
    card.style.backgroundColor = styleInfo.bg;
    card.style.borderColor = styleInfo.border;
    if (isActive) {
      card.style.borderLeft = `4px solid ${styleInfo.textColor}`;
    }

    card.innerHTML = `
      <div class="inst-top">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="inst-rank">#${index + 1}</span>
          <span class="inst-name">${intel.name}</span>
          <span class="badge" style="font-size: 9px; padding: 2px 6px;">${intel.family.split(' ')[0]}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
          <span class="inst-hover-hint">Intel &rarr;</span>
          <span class="inst-score ${isActive ? '' : 'inactive'}" style="color: ${styleInfo.textColor};">
            ${score.toFixed(3)} (${pct}%)
          </span>
        </div>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${pct}%; background: ${styleInfo.barGradient};"></div>
      </div>
    `;

    // Click on instrument card opens Dossier
    card.addEventListener('click', () => {
      openInstrumentDossier(inst, score, index + 1);
    });

    container.appendChild(card);
  });

  // Update summary metrics
  document.getElementById('metric-active-count').textContent = `${activeCount} / 18`;
  if (topInstrument) {
    const topIntel = INSTRUMENT_INTEL[topInstrument[0]] || { name: topInstrument[0] };
    document.getElementById('metric-top-instrument').textContent = topIntel.name;
    document.getElementById('metric-top-score').textContent = `Dominant Signal (${Math.round(topInstrument[1] * 100)}%)`;
  }
  document.getElementById('metric-duration').textContent = `${data.duration_seconds || 10.0}s`;

  // Spectrogram Image
  const specImg = document.getElementById('spectrogram-image');
  if (data.spectrogram_base64) {
    specImg.src = `data:image/png;base64,${data.spectrogram_base64}`;
    specImg.style.display = 'block';
  } else {
    specImg.style.display = 'none';
  }

  // Update the benchmark taxonomy cards with live scores and color intensity
  updateTaxonomyCards(predictions);

  // If user had selected "By Detected %", sort the taxonomy grid
  if (currentSortMode === 'prob') {
    sortTaxonomyGrid('prob');
  }
}

/**
 * Updates the 18 benchmark taxonomy cards with detected percentages and color intensity.
 */
function updateTaxonomyCards(predictions) {
  const taxonomyCards = document.querySelectorAll('#taxonomy-grid .inst-card');
  const sortedEntries = Object.entries(predictions).sort((a, b) => b[1] - a[1]);

  taxonomyCards.forEach(card => {
    const instKey = card.dataset.instrument;
    const score = predictions[instKey];
    if (score !== undefined) {
      card.dataset.score = score;
      const pct = Math.round(score * 100);
      const styleInfo = getColorIntensity(score);
      const rank = sortedEntries.findIndex(e => e[0] === instKey) + 1;

      card.style.backgroundColor = styleInfo.bg;
      card.style.borderColor = styleInfo.border;

      // Check if rank chip already exists, if not add it
      let rankEl = card.querySelector('.inst-rank');
      if (!rankEl) {
        rankEl = document.createElement('span');
        rankEl.className = 'inst-rank';
        card.querySelector('.inst-top').prepend(rankEl);
      }
      rankEl.textContent = `#${rank}`;

      // Update badge to display score
      let scoreBadge = card.querySelector('.badge');
      if (scoreBadge) {
        const intel = INSTRUMENT_INTEL[instKey];
        scoreBadge.textContent = `${intel ? intel.family.split(' ')[0] : 'Instrument'} &bull; ${pct}%`;
        scoreBadge.style.cssText = styleInfo.badgeStyle;
      }
    }
  });
}

/**
 * Sorts the 18 benchmark cards either alphabetically (A->Z) or by predicted score.
 */
function sortTaxonomyGrid(mode) {
  const grid = document.getElementById('taxonomy-grid');
  const cards = Array.from(grid.querySelectorAll('.inst-card'));

  cards.sort((a, b) => {
    if (mode === 'prob') {
      const scoreA = parseFloat(a.dataset.score || 0);
      const scoreB = parseFloat(b.dataset.score || 0);
      return scoreB - scoreA;
    } else {
      const nameA = a.querySelector('.inst-name').textContent.toLowerCase();
      const nameB = b.querySelector('.inst-name').textContent.toLowerCase();
      return nameA.localeCompare(nameB);
    }
  });

  // Re-append in new order
  cards.forEach(card => grid.appendChild(card));
}

/**
 * Dynamic slider update to toggle active/inactive states without full re-render.
 */
function updateActiveThreshold(thresholdVal) {
  const container = document.getElementById('instruments-container');
  const cards = container.querySelectorAll('.inst-card');

  let activeCount = 0;

  cards.forEach(card => {
    const score = parseFloat(card.dataset.score);
    const scoreSpan = card.querySelector('.inst-score');
    const styleInfo = getColorIntensity(score);

    if (score >= thresholdVal) {
      card.className = 'inst-card active';
      card.style.borderLeft = `4px solid ${styleInfo.textColor}`;
      scoreSpan.className = 'inst-score';
      activeCount++;
    } else {
      card.className = 'inst-card inactive';
      card.style.borderLeft = `1px solid ${styleInfo.border}`;
      scoreSpan.className = 'inst-score inactive';
    }
  });

  document.getElementById('metric-active-count').textContent = `${activeCount} / 18`;
}

/**
 * Opens the rich Multi-Layer Instrument Dossier modal.
 */
function openInstrumentDossier(instKey, score = null, rank = null) {
  const intel = INSTRUMENT_INTEL[instKey];
  if (!intel) return;

  currentModalInstrument = instKey;

  const modal = document.getElementById('instrument-modal');
  const nameEl = document.getElementById('modal-inst-name');
  const familyEl = document.getElementById('modal-inst-family');
  const soundEl = document.getElementById('modal-daily-sound');
  const spotEl = document.getElementById('modal-spot-mix');
  const cueEl = document.getElementById('modal-famous-cue');

  const chipAttack = document.getElementById('modal-chip-attack');
  const chipFeel = document.getElementById('modal-chip-feel');
  const chipFreq = document.getElementById('modal-chip-freq');

  const freqEl = document.getElementById('modal-frequency');
  const timbreEl = document.getElementById('modal-timbre');
  const originEl = document.getElementById('modal-origin');
  const contextEl = document.getElementById('modal-context');
  const liveStatusCard = document.getElementById('modal-live-status');

  nameEl.textContent = intel.name;
  familyEl.textContent = intel.family;

  // Fingerprint chips
  if (chipAttack) chipAttack.textContent = intel.attack || 'Acoustic Attack';
  if (chipFeel) chipFeel.textContent = intel.feel || 'Resonant';
  if (chipFreq) chipFreq.textContent = intel.frequency.split('–')[0].trim() || 'Variable';

  // 3-Layer Daily Life Sound Breakdown
  soundEl.textContent = `“${intel.dailyLifeSound}”`;
  if (spotEl) spotEl.textContent = intel.spotInSong;
  if (cueEl) cueEl.textContent = intel.famousCue;

  // Technical Specs
  freqEl.textContent = intel.frequency;
  timbreEl.textContent = intel.timbre;
  originEl.textContent = intel.origin;
  contextEl.textContent = intel.famousContext;

  // Live Track Inference section
  if (score !== null && score !== undefined) {
    const pct = Math.round(score * 100);
    const currentThreshold = parseFloat(document.getElementById('threshold-slider').value);
    const isActive = score >= currentThreshold;
    const styleInfo = getColorIntensity(score);

    liveStatusCard.style.display = 'flex';
    document.getElementById('modal-prediction-pct').textContent = `${score.toFixed(3)} (${pct}%)`;
    document.getElementById('modal-prediction-pct').style.color = styleInfo.textColor;

    const rankEl = document.getElementById('modal-prediction-rank');
    if (rank) {
      rankEl.textContent = `Mix Rank #${rank} of 18`;
    } else {
      rankEl.textContent = 'Mix Presence';
    }

    const badge = document.getElementById('modal-prediction-badge');
    badge.textContent = isActive ? 'Detected in Mix' : 'Below Cutoff';
    badge.className = `badge ${isActive ? 'badge-coral' : ''}`;

    const bar = document.getElementById('modal-prediction-bar');
    bar.style.width = `${pct}%`;
    bar.style.background = styleInfo.barGradient;
  } else {
    liveStatusCard.style.display = 'none';
  }

  modal.classList.add('open');
  modal.setAttribute('aria-hidden', 'false');
}

/**
 * Closes the Instrument Dossier modal.
 */
function closeInstrumentDossier() {
  const modal = document.getElementById('instrument-modal');
  if (modal) {
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
  }
  currentModalInstrument = null;
}
