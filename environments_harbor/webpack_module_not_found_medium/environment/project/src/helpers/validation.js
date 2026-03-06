function validateForm(form) {
  if (!form || typeof form !== 'object') {
    return false;
  }
  return Object.values(form).every(field => field && field.toString().trim() !== '');
}

function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

export { validateForm, validateEmail };