import { interval, Subject, combineLatest, merge, fromEvent } from 'rxjs';
import { map, filter, distinctUntilChanged, debounceTime, switchMap } from 'rxjs/operators';

interface WeatherData {
  temperature: number;
  humidity: number;
  pressure: number;
  windSpeed: number;
  timestamp: Date;
}

interface SensorReading {
  sensorId: string;
  value: number;
  unit: string;
}

interface AggregatedData {
  avgTemperature: number;
  avgHumidity: number;
  avgPressure: number;
  sampleCount: number;
}

export class WeatherService {
  private temperatureSubject$ = new Subject<number>();
  private humiditySubject$ = new Subject<number>();
  private pressureSubject$ = new Subject<number>();
  private windSpeedSubject$ = new Subject<number>();
  private sensorDataSubject$ = new Subject<SensorReading>();
  
  private dataBuffer: WeatherData[] = [];
  private aggregatedResults: AggregatedData[] = [];

  constructor() {
    console.log('WeatherService initialized');
  }

  public startMonitoring(): void {
    console.log('Starting weather monitoring...');
    
    // Issue 1: Temperature monitoring subscription never cleaned up
    interval(1000).pipe(
      map(() => this.generateTemperature())
    ).subscribe(temp => {
      this.temperatureSubject$.next(temp);
      console.log(`Temperature: ${temp}°C`);
    });

    // Issue 2: Humidity and pressure monitoring without cleanup
    interval(1500).pipe(
      map(() => this.generateHumidity())
    ).subscribe(humidity => {
      this.humiditySubject$.next(humidity);
      console.log(`Humidity: ${humidity}%`);
    });

    interval(2000).pipe(
      map(() => this.generatePressure())
    ).subscribe(pressure => {
      this.pressureSubject$.next(pressure);
      console.log(`Pressure: ${pressure} hPa`);
    });

    this.processWeatherData();
    this.aggregateData();
  }

  public processWeatherData(): void {
    // Issue 3: Combined weather data subscription without cleanup
    combineLatest([
      this.temperatureSubject$,
      this.humiditySubject$,
      this.pressureSubject$
    ]).pipe(
      debounceTime(500),
      map(([temp, humidity, pressure]) => ({
        temperature: temp,
        humidity: humidity,
        pressure: pressure,
        windSpeed: this.generateWindSpeed(),
        timestamp: new Date()
      }))
    ).subscribe(weatherData => {
      this.dataBuffer.push(weatherData);
      this.processBuffer();
    });

    // Additional stream merging for alerts
    merge(
      this.temperatureSubject$.pipe(filter(t => t > 35)),
      this.humiditySubject$.pipe(filter(h => h > 80))
    ).subscribe(value => {
      console.log(`Alert: Extreme value detected: ${value}`);
      this.sendAlert(value);
    });
  }

  public aggregateData(): void {
    interval(5000).subscribe(() => {
      if (this.dataBuffer.length > 0) {
        const aggregated = this.calculateAggregates();
        this.aggregatedResults.push(aggregated);
        console.log('Aggregated data:', aggregated);
      }
    });
  }

  public monitorSensorNetwork(sensorIds: string[]): void {
    // Issue 4: Loop creating multiple subscriptions without cleanup
    sensorIds.forEach(sensorId => {
      interval(3000).pipe(
        map(() => ({
          sensorId: sensorId,
          value: Math.random() * 100,
          unit: 'units'
        }))
      ).subscribe(reading => {
        this.sensorDataSubject$.next(reading);
        console.log(`Sensor ${reading.sensorId}: ${reading.value}`);
      });
    });

    // Process sensor data stream
    this.sensorDataSubject$.pipe(
      distinctUntilChanged((a, b) => a.sensorId === b.sensorId && Math.abs(a.value - b.value) < 0.1)
    ).subscribe(reading => {
      this.processSensorReading(reading);
    });
  }

  public enableRealTimeUpdates(): void {
    // Issue 5: Wind speed monitoring without cleanup
    interval(1800).pipe(
      map(() => this.generateWindSpeed())
    ).subscribe(windSpeed => {
      this.windSpeedSubject$.next(windSpeed);
      console.log(`Wind Speed: ${windSpeed} km/h`);
    });

    // Combined monitoring for severe weather detection
    combineLatest([
      this.windSpeedSubject$,
      this.pressureSubject$,
      this.temperatureSubject$
    ]).pipe(
      filter(([wind, pressure, temp]) => wind > 50 || pressure < 980 || temp > 40)
    ).subscribe(([wind, pressure, temp]) => {
      console.log(`Severe weather conditions: Wind=${wind}, Pressure=${pressure}, Temp=${temp}`);
      this.handleSevereWeather(wind, pressure, temp);
    });
  }

  private processBuffer(): void {
    if (this.dataBuffer.length > 100) {
      this.dataBuffer = this.dataBuffer.slice(-100);
    }
  }

  private calculateAggregates(): AggregatedData {
    const count = this.dataBuffer.length;
    const sum = this.dataBuffer.reduce(
      (acc, data) => ({
        temperature: acc.temperature + data.temperature,
        humidity: acc.humidity + data.humidity,
        pressure: acc.pressure + data.pressure
      }),
      { temperature: 0, humidity: 0, pressure: 0 }
    );

    return {
      avgTemperature: sum.temperature / count,
      avgHumidity: sum.humidity / count,
      avgPressure: sum.pressure / count,
      sampleCount: count
    };
  }

  private generateTemperature(): number {
    return Math.random() * 40 + 10;
  }

  private generateHumidity(): number {
    return Math.random() * 100;
  }

  private generatePressure(): number {
    return Math.random() * 50 + 980;
  }

  private generateWindSpeed(): number {
    return Math.random() * 80;
  }

  private processSensorReading(reading: SensorReading): void {
    console.log(`Processing sensor reading from ${reading.sensorId}`);
  }

  private sendAlert(value: number): void {
    console.log(`Sending alert for value: ${value}`);
  }

  private handleSevereWeather(wind: number, pressure: number, temp: number): void {
    console.log(`Handling severe weather event`);
  }

  public getAggregatedResults(): AggregatedData[] {
    return this.aggregatedResults;
  }

  public getBufferSize(): number {
    return this.dataBuffer.length;
  }
}