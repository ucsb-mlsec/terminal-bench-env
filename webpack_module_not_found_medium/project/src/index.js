import Button from '@components/Button';
import Card from '@components/Card';
import { formatDate, parseDate } from '@utils/dateUtils';
import { capitalize } from '@utils/stringUtils';
import { validateForm } from '@helpers/validation';
import { processData } from '@helpers/dataProcessor';

export function init() {
  console.log('Application initialized');
  const today = formatDate(new Date());
  console.log('Today is:', today);
  
  const button = new Button('Submit');
  const card = new Card({ title: 'Welcome' });
  
  console.log('Components loaded:', button, card);
  console.log('Validation ready:', validateForm);
  console.log('Data processor ready:', processData);
}