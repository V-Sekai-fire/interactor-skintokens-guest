// Only one rule: every block has braces. The rest of the JavaScript style stays as it is.
export default [
  {
    files: ['**/*.js', '**/*.mjs', '**/*.cjs'],
    rules: {
      curly: ['error', 'all'],
    },
  },
];
