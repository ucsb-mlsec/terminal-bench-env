<?php

namespace App;

class User
{
    protected array $fillable = ['name', 'email', 'bio', 'avatar_url', 'timezone'];
    
    protected array $guarded = ['password', 'remember_token', 'is_admin', 'email_verified_at'];
    
    protected array $attributes = [];
    
    public function __construct()
    {
        $this->attributes = [];
    }
    
    public function fill(array $data)
    {
        foreach ($data as $key => $value) {
            if (!empty($this->fillable)) {
                if (in_array($key, $this->fillable)) {
                    $this->attributes[$key] = $value;
                }
            } elseif (!empty($this->guarded)) {
                if (!in_array($key, $this->guarded)) {
                    $this->attributes[$key] = $value;
                }
            } else {
                $this->attributes[$key] = $value;
            }
        }
        
        return $this;
    }
    
    public function get(string $key)
    {
        return $this->attributes[$key] ?? null;
    }
    
    public function toArray(): array
    {
        return $this->attributes;
    }
}