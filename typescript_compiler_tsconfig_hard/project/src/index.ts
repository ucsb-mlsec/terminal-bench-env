import { helper } from '@shared/utils';
import { Config } from '@shared/types';

function main(): void {
  const appConfig: Config = {
    name: 'MyApplication',
    port: 3000
  };

  console.log('Starting application with config:', appConfig);

  const result = helper('test');
  console.log('Helper function result:', result);

  console.log(`Server will run on port ${appConfig.port}`);
}

main();