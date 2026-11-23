/**
 * Chaturanga piece set: inline SVG drawn for the historical pieces
 * (chariot, elephant, horse, minister, king, foot soldier) instead of
 * Western chess glyphs. Shapes use currentColor for the body and the
 * --piece-line CSS variable for outlines, so .white-piece / .black-piece
 * set the colours in style.css.
 */

// One shared pedestal so every piece stands on the same base
const PIECE_BASE = '<path class="pc-base" d="M24 90h52a4 4 0 0 0 0-8H24a4 4 0 0 0 0 8z"/>';

const PIECE_SHAPES = {
  // Padati: foot soldier with a spear
  1: `<path d="M64 18 70 32H58z" class="pc-detail"/>
      <path d="M62.5 30h3v52h-3z" class="pc-detail"/>
      <path d="M33 82c0-14 5-24 17-27-6-3-9-8-9-13a9 9 0 0 1 18 0c0 5-3 10-9 13 12 3 17 13 17 27z"/>`,

  // Ashva: horse head
  2: `<path d="M31 82c1-11 7-18 13-24-6 1-11 4-15 0-4-3-3-8 1-12l13-12c2-6 6-11 12-13l1-7 6 8c11 5 17 17 15 32-1 9-5 17-5 28z"/>
      <circle cx="49" cy="32" r="2.6" class="pc-eye"/>
      <path d="M56 26c6 5 9 13 8 24" class="pc-line"/>`,

  // Gaja: elephant head with ears, tusks and trunk
  3: `<path d="M38 27C24 22 12 32 14 47c2 12 12 18 24 16z"/>
      <path d="M62 27c14-5 26 5 24 20-2 12-12 18-24 16z"/>
      <path d="M50 18c13 0 21 10 21 22s-8 22-21 22-21-10-21-22 8-22 21-22z"/>
      <path d="M50 56v13c0 9 9 11 13 5" class="pc-trunk-o"/>
      <path d="M50 56v13c0 9 9 11 13 5" class="pc-trunk"/>
      <path d="M41 58c-5 3-8 9-7 15" class="pc-tusk"/>
      <path d="M59 58c4 3 6 7 6 11" class="pc-tusk"/>
      <path d="M50 24l5 6-5 6-5-6z" class="pc-detail"/>
      <circle cx="42" cy="42" r="2.6" class="pc-eye"/>
      <circle cx="58" cy="42" r="2.6" class="pc-eye"/>`,

  // Ratha: chariot with canopy, platform and a spoked wheel
  4: `<path d="M30 22h40l-6 10H36z"/>
      <path d="M38 32h3v12h-3zM59 32h3v12h-3z" class="pc-detail"/>
      <path d="M26 44h48v12H26z"/>
      <circle cx="50" cy="68" r="14"/>
      <circle cx="50" cy="68" r="4" class="pc-hub"/>
      <path d="M50 55v26M37 68h26M41 59l18 18M59 59 41 77" class="pc-line"/>`,

  // Mantri: minister in a turban with a plume
  5: `<path d="M50 10c3 4 3 8 0 11-3-3-3-7 0-11z" class="pc-detail"/>
      <path d="M36 34c0-9 6-14 14-14s14 5 14 14z"/>
      <path d="M36 32h28" class="pc-line"/>
      <circle cx="50" cy="42" r="9"/>
      <path d="M31 82c0-18 7-28 19-30 12 2 19 12 19 30z"/>
      <path d="M50 54v26" class="pc-line"/>`,

  // Raja: king with a crown, tallest piece
  6: `<path d="M50 6l3 6h-6z" class="pc-detail"/>
      <path d="M34 30 32 14l9 8 9-11 9 11 9-8-2 16z"/>
      <path d="M34 30h32" class="pc-line"/>
      <circle cx="50" cy="40" r="9"/>
      <path d="M28 82c0-19 8-30 22-32 14 2 22 13 22 32z"/>
      <path d="M40 64h20" class="pc-line"/>
      <circle cx="50" cy="19" r="2.2" class="pc-hub"/>`,
};

const PIECE_NAMES_UI = {
  1: 'Padati (foot soldier)',
  2: 'Ashva (horse)',
  3: 'Gaja (elephant)',
  4: 'Ratha (chariot)',
  5: 'Mantri (minister)',
  6: 'Raja (king)',
};

/** SVG markup for a piece code (+ White, - Black). */
function pieceSVG(code) {
  const shape = PIECE_SHAPES[Math.abs(code)];
  if (!shape) return '';
  const side = code > 0 ? 'white-piece' : 'black-piece';
  const who = code > 0 ? 'White' : 'Black';
  return `<svg class="piece-svg ${side}" viewBox="0 0 100 100" role="img"
               aria-label="${who} ${PIECE_NAMES_UI[Math.abs(code)]}">${shape}${PIECE_BASE}</svg>`;
}
